def sample_data():
    """Create a simple dataset for testing"""
    X, y = make_classification(
        n_samples=1000, n_features=20, n_classes=2, random_state=42
    )
    X = StandardScaler().fit_transform(X)
    return train_test_split(X, y, test_size=0.2, random_state=42)