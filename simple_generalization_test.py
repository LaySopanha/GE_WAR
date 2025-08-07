# simple_generalization_test.py - Quick generalization assessment
import numpy as np
import json

def analyze_existing_models():
    """Analyze existing saved models to assess generalization"""
    
    print("🔍 ANALYZING EXISTING MODEL RESULTS FOR GENERALIZATION")
    print("="*70)
    
    # Load all your saved model configs
    model_configs = []
    model_files = [
        'ge0_config_run_1jcrispy_fold_0_ntge_97397.npy',
        'ge0_config_run_7enydxkm_fold_0_ntge_97566.npy', 
        'ge0_config_run_3qh1fr41_fold_0_ntge_97819.npy',
        'ge0_config_run_d1o4uye4_fold_0_ntge_97871.npy',
        'ge0_config_run_3nf9jwii_fold_0_ntge_97972.npy',
        'ge0_config_run_40ioiqmo_fold_0_ntge_98623.npy',
        'ge0_config_run_5k5plsz9_fold_0_ntge_98888.npy',
        'ge0_config_run_o3p9mc6i_fold_3_ntge_99917.npy',
        'ge0_config_run_clwpanu7_fold_0_ntge_99942.npy',
        'ge0_config_run_mxr9qy07_fold_0_ntge_99962.npy'
    ]
    
    print("📊 Loading model configurations...")
    for model_file in model_files:
        try:
            config = np.load(model_file, allow_pickle=True).item()
            ntge = int(model_file.split('_ntge_')[1].split('.')[0])
            model_configs.append({
                'file': model_file,
                'ntge': ntge,
                'config': config
            })
            print(f"  ✅ {model_file}: NTGE={ntge}")
        except Exception as e:
            print(f"  ❌ Failed to load {model_file}: {e}")
    
    if len(model_configs) == 0:
        print("❌ No model configs found!")
        return
    
    # Analyze hyperparameter consistency
    print(f"\n🔬 HYPERPARAMETER ANALYSIS ({len(model_configs)} models)")
    print("-" * 50)
    
    # Extract key hyperparameters
    dropout_rates = [m['config']['dropout_rate'] for m in model_configs]
    noise_levels = [m['config']['noise_level'] for m in model_configs]
    learning_rates = [m['config']['lr'] for m in model_configs]
    ntge_values = [m['ntge'] for m in model_configs]
    
    # Statistical analysis
    stats = {
        'dropout_rate': {
            'mean': np.mean(dropout_rates),
            'std': np.std(dropout_rates),
            'range': (np.min(dropout_rates), np.max(dropout_rates))
        },
        'noise_level': {
            'mean': np.mean(noise_levels),
            'std': np.std(noise_levels),
            'range': (np.min(noise_levels), np.max(noise_levels))
        },
        'learning_rate': {
            'mean': np.mean(learning_rates),
            'std': np.std(learning_rates),
            'range': (np.min(learning_rates), np.max(learning_rates))
        },
        'ntge': {
            'mean': np.mean(ntge_values),
            'std': np.std(ntge_values),
            'range': (np.min(ntge_values), np.max(ntge_values))
        }
    }
    
    print("📈 Hyperparameter Distributions:")
    for param, stat in stats.items():
        cv = (stat['std'] / stat['mean']) * 100 if stat['mean'] > 0 else 0
        print(f"  {param}:")
        print(f"    Mean: {stat['mean']:.6f}")
        print(f"    Std:  {stat['std']:.6f}")
        print(f"    Range: {stat['range'][0]:.6f} - {stat['range'][1]:.6f}")
        print(f"    CV: {cv:.1f}%")
        print()
    
    # Correlation analysis
    print("🔗 CORRELATION ANALYSIS:")
    print("-" * 30)
    
    # Correlation between hyperparameters and NTGE
    correlations = {}
    for param in ['dropout_rate', 'noise_level', 'learning_rate']:
        values = [m['config'][param] for m in model_configs]
        corr = np.corrcoef(values, ntge_values)[0, 1]
        correlations[param] = corr
        direction = "↗️ Higher" if corr > 0 else "↘️ Lower" if corr < 0 else "➡️ No"
        strength = "Strong" if abs(corr) > 0.7 else "Moderate" if abs(corr) > 0.3 else "Weak"
        print(f"  {param} vs NTGE: {corr:.3f} ({strength}, {direction} correlation)")
    
    # Generalization assessment
    print(f"\n🎯 GENERALIZATION ASSESSMENT:")
    print("=" * 40)
    
    ntge_cv = (stats['ntge']['std'] / stats['ntge']['mean']) * 100
    best_ntge = min(ntge_values)
    worst_ntge = max(ntge_values)
    variance_ratio = worst_ntge / best_ntge
    
    print(f"📊 NTGE Performance:")
    print(f"  Best: {best_ntge:,}")
    print(f"  Worst: {worst_ntge:,}")
    print(f"  Mean: {stats['ntge']['mean']:.0f}")
    print(f"  Std: {stats['ntge']['std']:.0f}")
    print(f"  CV: {ntge_cv:.1f}%")
    print(f"  Variance Ratio: {variance_ratio:.2f}x")
    
    # Hyperparameter stability
    hyper_stability = []
    for param in ['dropout_rate', 'noise_level', 'learning_rate']:
        cv = (stats[param]['std'] / stats[param]['mean']) * 100 if stats[param]['mean'] > 0 else 100
        hyper_stability.append(cv)
    
    avg_hyper_cv = np.mean(hyper_stability)
    
    print(f"\n🔮 GENERALIZATION PREDICTION:")
    print("-" * 35)
    
    # Scoring system
    score = 0
    reasons = []
    
    # NTGE consistency (40% weight)
    if ntge_cv < 5:
        score += 40
        reasons.append("✅ Excellent NTGE consistency (<5% CV)")
    elif ntge_cv < 10:
        score += 30
        reasons.append("🟡 Good NTGE consistency (<10% CV)")
    elif ntge_cv < 20:
        score += 20
        reasons.append("⚠️ Moderate NTGE consistency (<20% CV)")
    else:
        score += 0
        reasons.append("❌ Poor NTGE consistency (>20% CV)")
    
    # Hyperparameter stability (30% weight)
    if avg_hyper_cv < 15:
        score += 30
        reasons.append("✅ Stable hyperparameters")
    elif avg_hyper_cv < 30:
        score += 20
        reasons.append("🟡 Moderately stable hyperparameters")
    else:
        score += 10
        reasons.append("⚠️ Unstable hyperparameters")
    
    # Performance level (30% weight)
    if best_ntge < 50000:
        score += 30
        reasons.append("✅ Excellent best performance (<50K NTGE)")
    elif best_ntge < 80000:
        score += 20
        reasons.append("🟡 Good best performance (<80K NTGE)")
    elif best_ntge < 100000:
        score += 10
        reasons.append("⚠️ Moderate best performance (<100K NTGE)")
    else:
        score += 0
        reasons.append("❌ Poor best performance (>100K NTGE)")
    
    # Final assessment
    if score >= 80:
        assessment = "🟢 EXCELLENT"
        confidence = "Very High"
        expected_multiplier = (1.0, 1.3)
    elif score >= 60:
        assessment = "🟡 GOOD"
        confidence = "High"
        expected_multiplier = (0.8, 1.6)
    elif score >= 40:
        assessment = "🟠 MODERATE"
        confidence = "Medium"
        expected_multiplier = (0.6, 2.0)
    else:
        assessment = "🔴 CONCERNING"
        confidence = "Low"
        expected_multiplier = (0.4, 2.5)
    
    print(f"Overall Score: {score}/100")
    print(f"Assessment: {assessment}")
    print(f"Confidence: {confidence}")
    print()
    
    print("Reasoning:")
    for reason in reasons:
        print(f"  {reason}")
    
    print(f"\n📈 EXPECTED PRIVATE DATASET PERFORMANCE:")
    print("-" * 45)
    expected_range = (
        int(best_ntge * expected_multiplier[0]),
        int(best_ntge * expected_multiplier[1])
    )
    print(f"Expected NTGE Range: {expected_range[0]:,} - {expected_range[1]:,}")
    print(f"Most Likely NTGE: ~{int(best_ntge * np.mean(expected_multiplier)):,}")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    print("-" * 20)
    
    if score >= 70:
        print("✅ Proceed with current best model for submission")
        print("✅ Consider ensemble of top 3 models for robustness")
    elif score >= 50:
        print("🟡 Current model is decent but could be improved")
        print("🟡 Run refined sweep with narrower hyperparameter ranges")
        print("🟡 Consider ensemble approach")
    else:
        print("🔴 High risk of poor generalization")
        print("🔴 Strongly recommend refined hyperparameter search")
        print("🔴 Focus on more conservative/robust configurations")
    
    return {
        'score': score,
        'assessment': assessment,
        'expected_range': expected_range,
        'best_ntge': best_ntge,
        'stats': stats
    }

if __name__ == "__main__":
    analyze_existing_models()
