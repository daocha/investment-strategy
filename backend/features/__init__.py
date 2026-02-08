from backend.features.technical import calculate_technical_features
from backend.features.market import calculate_market_features

class FeaturesPipeline:
    """
    Central pipeline for feature engineering.
    Ensures all functional feature groups are applied in sequence.
    """
    @staticmethod
    def generate_feature_set(df):
        """
        Unified entry point to generate all features required for prediction.
        """
        df = df.copy()
        
        # 1. Apply Technical Indicators
        df = calculate_technical_features(df)
        
        # 2. Apply Market Features
        df = calculate_market_features(df)
        
        # 3. Final Cleanup
        df = df.bfill()
        
        return df

# Legacy wrapper for backward compatibility during transition
def add_technical_indicators(df):
    return FeaturesPipeline.generate_feature_set(df)
