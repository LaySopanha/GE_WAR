import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, in_channels // 8, 1, bias=False)
        self.conv2 = nn.Conv1d(in_channels // 8, 1, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        attn = self.sigmoid(self.conv2(F.relu(self.conv1(x))))
        return x * attn, attn

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Sequential(
            nn.Conv1d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv1d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        attn = self.sigmoid(self.fc(self.avg_pool(x)) + self.fc(self.max_pool(x)))
        return x * attn, attn

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, use_attention=True):
        super(CNNBlock, self).__init__()
        self.use_attention = use_attention
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        if use_attention:
            self.spatial_attention = SpatialAttention(out_channels)
            self.channel_attention = ChannelAttention(out_channels)
        self.pool = nn.MaxPool1d(2, 2)

    def forward(self, x):
        out = self.relu(self.bn(self.conv(x)))
        spatial_weights, channel_weights = None, None
        if self.use_attention:
            out, channel_weights = self.channel_attention(out)
            out, spatial_weights = self.spatial_attention(out)
        return self.pool(out), spatial_weights, channel_weights

class CNN_LSTM_SCA(nn.Module):
    def __init__(self, config, num_sample_pts, classes):
        super(CNN_LSTM_SCA, self).__init__()
        self.cnn_layers = nn.ModuleList()
        in_channels = 1
        # CNN Part
        for i in range(len(config['cnn_channels'])):
            out_channels = config['cnn_channels'][i]
            kernel_size = config['cnn_kernels'][i]
            self.cnn_layers.append(CNNBlock(in_channels, out_channels, kernel_size=kernel_size, use_attention=config.get('use_attention', True)))
            in_channels = out_channels
        
        # Calculate LSTM input size dynamically
        with torch.no_grad():
            dummy_input = torch.randn(1, 1, num_sample_pts)
            cnn_out = dummy_input
            for layer in self.cnn_layers:
                cnn_out, _, _ = layer(cnn_out)
            lstm_input_size = cnn_out.size(1)

        # LSTM Part
        lstm_output_size = config['lstm_hidden_size'] * (2 if config.get('bidirectional', False) else 1)
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=config['lstm_hidden_size'],
            num_layers=config['lstm_num_layers'],
            batch_first=True,
            dropout=config.get('lstm_dropout', 0.1) if config['lstm_num_layers'] > 1 else 0,
            bidirectional=config.get('bidirectional', False)
        )
        # Attention over LSTM outputs
        self.lstm_attention = nn.Sequential(
            nn.Linear(lstm_output_size, lstm_output_size // 2),
            nn.Tanh(),
            nn.Linear(lstm_output_size // 2, 1)
        )

        # Classifier Part
        self.classifier = nn.Sequential(
            nn.Dropout(config.get('dropout', 0.5)),
            nn.Linear(lstm_output_size, config.get('fc_hidden', 128)),
            nn.ReLU(inplace=True),
            nn.Dropout(config.get('dropout', 0.5)),
            nn.Linear(config.get('fc_hidden', 128), classes)
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # CNN Feature Extraction
        features = x
        for layer in self.cnn_layers:
            features, _, _ = layer(features)
        
        # Prepare for LSTM
        features = features.transpose(1, 2)  # (N, C, L) -> (N, L, C)
        
        # LSTM + Attention
        lstm_out, _ = self.lstm(features)
        attention_weights = F.softmax(self.lstm_attention(lstm_out), dim=1)
        attended_lstm = torch.sum(lstm_out * attention_weights, dim=1)
        
        # Classifier
        output = self.classifier(attended_lstm)
        return output

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding=kernel_size//2, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, 1, padding=kernel_size//2, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class ResNetSCA(nn.Module):
    def __init__(self, config, num_sample_pts, classes):
        super(ResNetSCA, self).__init__()
        self.in_channels = config['initial_filters']
        
        self.conv1 = nn.Conv1d(1, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        self.layers = nn.ModuleList()
        for block_config in config['blocks']:
            num_blocks, out_channels, stride = block_config
            self.layers.append(self._make_layer(num_blocks, out_channels, stride))
            
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        
        with torch.no_grad():
            dummy_input = torch.randn(1, 1, num_sample_pts)
            features = self.maxpool(self.relu(self.bn1(self.conv1(dummy_input))))
            for layer in self.layers:
                features = layer(features)
            fc_input_size = features.size(1)

        self.fc = nn.Linear(fc_input_size, classes)

    def _make_layer(self, num_blocks, out_channels, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(ResidualBlock(self.in_channels, out_channels, stride=stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        for layer in self.layers:
            out = layer(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out
