#!/usr/bin/env python3
"""
Feature Flag System for FLUX-Sci-Lang
Enables gradual rollout and A/B testing of new features
"""

import json
import random
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from enum import Enum


class RolloutStrategy(Enum):
    """Feature rollout strategies"""
    PERCENTAGE = "percentage"
    USER_LIST = "user_list"
    GRADUAL = "gradual"
    RING = "ring"
    AB_TEST = "ab_test"


class Feature:
    """Represents a feature with flags"""

    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self.enabled = False
        self.strategy = RolloutStrategy.PERCENTAGE
        self.percentage = 0
        self.user_list = []
        self.rings = []
        self.variants = {}
        self.metadata = {}
        self.created_at = datetime.now()
        self.updated_at = datetime.now()

    def is_enabled_for(self, user_id: Optional[str] = None, context: Dict = None) -> bool:
        """Check if feature is enabled for specific user/context"""

        if not self.enabled:
            return False

        if self.strategy == RolloutStrategy.PERCENTAGE:
            return self._check_percentage(user_id)

        elif self.strategy == RolloutStrategy.USER_LIST:
            return user_id in self.user_list if user_id else False

        elif self.strategy == RolloutStrategy.GRADUAL:
            return self._check_gradual_rollout()

        elif self.strategy == RolloutStrategy.RING:
            return self._check_ring_deployment(user_id, context)

        elif self.strategy == RolloutStrategy.AB_TEST:
            return self._check_ab_test(user_id)

        return False

    def _check_percentage(self, user_id: Optional[str]) -> bool:
        """Check percentage-based rollout"""
        if self.percentage >= 100:
            return True
        if self.percentage <= 0:
            return False

        if user_id:
            # Consistent hashing for user
            hash_value = int(hashlib.md5(user_id.encode()).hexdigest(), 16)
            return (hash_value % 100) < self.percentage
        else:
            # Random for anonymous
            return random.random() * 100 < self.percentage

    def _check_gradual_rollout(self) -> bool:
        """Check time-based gradual rollout"""
        if 'start_date' not in self.metadata:
            return False

        start = datetime.fromisoformat(self.metadata['start_date'])
        end = datetime.fromisoformat(self.metadata.get('end_date', '2025-12-31'))
        now = datetime.now()

        if now < start:
            return False
        if now > end:
            return True

        # Calculate percentage based on time
        total_duration = (end - start).total_seconds()
        elapsed = (now - start).total_seconds()
        current_percentage = (elapsed / total_duration) * 100

        return random.random() * 100 < current_percentage

    def _check_ring_deployment(self, user_id: Optional[str], context: Dict) -> bool:
        """Check ring-based deployment"""
        if not self.rings:
            return False

        user_ring = context.get('ring', 'production') if context else 'production'

        for ring in self.rings:
            if ring['name'] == user_ring:
                return ring.get('enabled', False)

        return False

    def _check_ab_test(self, user_id: Optional[str]) -> str:
        """Get A/B test variant for user"""
        if not user_id or not self.variants:
            return 'control'

        # Consistent variant assignment
        hash_value = int(hashlib.md5(user_id.encode()).hexdigest(), 16)
        variant_index = hash_value % len(self.variants)

        return list(self.variants.keys())[variant_index]

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'name': self.name,
            'description': self.description,
            'enabled': self.enabled,
            'strategy': self.strategy.value,
            'percentage': self.percentage,
            'user_list': self.user_list,
            'rings': self.rings,
            'variants': self.variants,
            'metadata': self.metadata,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }


class FeatureFlagManager:
    """Manages feature flags for the platform"""

    def __init__(self, config_file: str = 'enhancements/feature_flags.json'):
        self.config_file = config_file
        self.features = {}
        self.load_features()

    def load_features(self):
        """Load features from configuration"""
        try:
            with open(self.config_file, 'r') as f:
                data = json.load(f)
                for name, config in data.get('features', {}).items():
                    feature = Feature(name, config.get('description', ''))
                    feature.enabled = config.get('enabled', False)
                    feature.strategy = RolloutStrategy(config.get('strategy', 'percentage'))
                    feature.percentage = config.get('percentage', 0)
                    feature.user_list = config.get('user_list', [])
                    feature.rings = config.get('rings', [])
                    feature.variants = config.get('variants', {})
                    feature.metadata = config.get('metadata', {})
                    self.features[name] = feature
        except FileNotFoundError:
            # Initialize with default features
            self._init_default_features()

    def save_features(self):
        """Save features to configuration"""
        data = {
            'version': '1.0.0',
            'features': {
                name: feature.to_dict()
                for name, feature in self.features.items()
            }
        }

        import os
        os.makedirs(os.path.dirname(self.config_file), exist_ok=True)

        with open(self.config_file, 'w') as f:
            json.dump(data, f, indent=2)

    def _init_default_features(self):
        """Initialize default feature flags"""

        # UI/UX Features
        dark_mode = Feature('dark_mode', 'Dark theme support')
        dark_mode.enabled = True
        dark_mode.percentage = 100
        self.features['dark_mode'] = dark_mode

        # Advanced solver features
        gpu_solver = Feature('gpu_acceleration', 'GPU-accelerated solving')
        gpu_solver.enabled = True
        gpu_solver.strategy = RolloutStrategy.PERCENTAGE
        gpu_solver.percentage = 50  # 50% rollout
        self.features['gpu_acceleration'] = gpu_solver

        # Collaboration
        collab = Feature('real_time_collaboration', 'Real-time collaborative solving')
        collab.enabled = True
        collab.strategy = RolloutStrategy.RING
        collab.rings = [
            {'name': 'internal', 'enabled': True},
            {'name': 'beta', 'enabled': True},
            {'name': 'production', 'enabled': False}
        ]
        self.features['real_time_collaboration'] = collab

        # ML features
        ml_prediction = Feature('ml_prediction', 'ML-powered solution prediction')
        ml_prediction.enabled = True
        ml_prediction.strategy = RolloutStrategy.AB_TEST
        ml_prediction.variants = {
            'control': {'model': None},
            'variant_a': {'model': 'pinn_v1'},
            'variant_b': {'model': 'deeponet_v1'}
        }
        self.features['ml_prediction'] = ml_prediction

        # New visualization
        webgl_3d = Feature('webgl_3d_viz', 'WebGL 3D visualization')
        webgl_3d.enabled = True
        webgl_3d.strategy = RolloutStrategy.GRADUAL
        webgl_3d.metadata = {
            'start_date': datetime.now().isoformat(),
            'end_date': (datetime.now() + timedelta(days=30)).isoformat()
        }
        self.features['webgl_3d_viz'] = webgl_3d

    def is_enabled(self, feature_name: str, user_id: Optional[str] = None,
                   context: Optional[Dict] = None) -> bool:
        """Check if a feature is enabled"""

        if feature_name not in self.features:
            return False

        feature = self.features[feature_name]
        return feature.is_enabled_for(user_id, context)

    def get_variant(self, feature_name: str, user_id: Optional[str] = None) -> str:
        """Get A/B test variant for a feature"""

        if feature_name not in self.features:
            return 'control'

        feature = self.features[feature_name]

        if feature.strategy != RolloutStrategy.AB_TEST:
            return 'control'

        return feature._check_ab_test(user_id)

    def enable_feature(self, feature_name: str, percentage: int = 100):
        """Enable a feature with given percentage"""

        if feature_name not in self.features:
            self.features[feature_name] = Feature(feature_name)

        feature = self.features[feature_name]
        feature.enabled = True
        feature.percentage = percentage
        feature.updated_at = datetime.now()
        self.save_features()

    def disable_feature(self, feature_name: str):
        """Disable a feature"""

        if feature_name in self.features:
            self.features[feature_name].enabled = False
            self.features[feature_name].updated_at = datetime.now()
            self.save_features()

    def set_rollout_percentage(self, feature_name: str, percentage: int):
        """Set rollout percentage for a feature"""

        if feature_name not in self.features:
            self.features[feature_name] = Feature(feature_name)

        feature = self.features[feature_name]
        feature.strategy = RolloutStrategy.PERCENTAGE
        feature.percentage = max(0, min(100, percentage))
        feature.updated_at = datetime.now()
        self.save_features()

    def add_user_to_feature(self, feature_name: str, user_id: str):
        """Add user to feature whitelist"""

        if feature_name not in self.features:
            self.features[feature_name] = Feature(feature_name)

        feature = self.features[feature_name]

        if user_id not in feature.user_list:
            feature.user_list.append(user_id)
            feature.updated_at = datetime.now()
            self.save_features()

    def get_all_features(self) -> Dict[str, Dict]:
        """Get all features and their status"""

        return {
            name: {
                'enabled': feature.enabled,
                'strategy': feature.strategy.value,
                'percentage': feature.percentage,
                'description': feature.description
            }
            for name, feature in self.features.items()
        }

    def get_user_features(self, user_id: str, context: Optional[Dict] = None) -> List[str]:
        """Get all enabled features for a user"""

        enabled_features = []

        for name, feature in self.features.items():
            if feature.is_enabled_for(user_id, context):
                enabled_features.append(name)

        return enabled_features


def create_feature_flag_middleware(app, flag_manager: FeatureFlagManager):
    """Create Flask middleware for feature flags"""

    @app.before_request
    def check_feature_flags():
        """Check feature flags before each request"""
        from flask import request, g

        # Get user ID from session or header
        user_id = None
        if hasattr(request, 'headers'):
            user_id = request.headers.get('X-User-ID')

        # Get context (could include user role, region, etc.)
        context = {
            'ring': request.headers.get('X-Deployment-Ring', 'production'),
            'region': request.headers.get('X-Region', 'us-east'),
            'client': request.headers.get('User-Agent', '')
        }

        # Store feature flags in Flask g object
        g.feature_flags = flag_manager
        g.user_id = user_id
        g.context = context
        g.enabled_features = flag_manager.get_user_features(user_id, context)

    @app.route('/api/features')
    def get_features():
        """Get enabled features for current user"""
        from flask import g, jsonify

        return jsonify({
            'user_id': g.user_id,
            'enabled_features': g.enabled_features,
            'all_features': flag_manager.get_all_features()
        })

    @app.route('/api/features/<feature_name>/check')
    def check_feature(feature_name):
        """Check if specific feature is enabled"""
        from flask import g, jsonify

        enabled = flag_manager.is_enabled(feature_name, g.user_id, g.context)
        variant = flag_manager.get_variant(feature_name, g.user_id)

        return jsonify({
            'feature': feature_name,
            'enabled': enabled,
            'variant': variant
        })

    return flag_manager


# Example usage in Flask app
def feature_required(feature_name):
    """Decorator to require feature flag"""

    def decorator(f):
        def wrapper(*args, **kwargs):
            from flask import g, jsonify

            if not g.feature_flags.is_enabled(feature_name, g.user_id, g.context):
                return jsonify({
                    'error': f'Feature {feature_name} is not enabled for this user'
                }), 403

            return f(*args, **kwargs)

        wrapper.__name__ = f.__name__
        return wrapper

    return decorator


if __name__ == "__main__":
    # Test feature flag system
    manager = FeatureFlagManager()

    # Test different scenarios
    print("Testing Feature Flags:")
    print("-" * 40)

    # Test percentage rollout
    print(f"GPU acceleration (50% rollout):")
    for i in range(5):
        user_id = f"user_{i}"
        enabled = manager.is_enabled('gpu_acceleration', user_id)
        print(f"  User {user_id}: {'✓' if enabled else '✗'}")

    # Test A/B test
    print(f"\nML Prediction A/B test:")
    for i in range(5):
        user_id = f"user_{i}"
        variant = manager.get_variant('ml_prediction', user_id)
        print(f"  User {user_id}: {variant}")

    # Test ring deployment
    print(f"\nCollaboration (ring deployment):")
    for ring in ['internal', 'beta', 'production']:
        context = {'ring': ring}
        enabled = manager.is_enabled('real_time_collaboration', 'test_user', context)
        print(f"  Ring {ring}: {'✓' if enabled else '✗'}")

    # Save configuration
    manager.save_features()
    print("\nFeature flags saved to configuration.")