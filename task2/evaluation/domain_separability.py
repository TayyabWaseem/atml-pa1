import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def compute_domain_separability(backbone, source_val_loaders, target_val_loader, device, seed=6304):
    """
    Computes domain separability score using a balanced logistic regression classifier.
    """
    backbone.eval()
    
    source_features = []
    target_features = []
    
    with torch.no_grad():
        for domain, loader in source_val_loaders.items():
            for x, _ in loader:
                x = x.to(device)
                feats = backbone(x)
                source_features.extend(feats.cpu().numpy())
                
        for x, _ in target_val_loader:
            x = x.to(device)
            feats = backbone(x)
            target_features.extend(feats.cpu().numpy())
            
    source_features = np.array(source_features)
    target_features = np.array(target_features)
    
    # Collect equal numbers of source-validation and target features
    min_len = min(len(source_features), len(target_features))
    
    # Randomly subsample to match lengths
    np.random.seed(seed)
    source_idx = np.random.choice(len(source_features), min_len, replace=False)
    target_idx = np.random.choice(len(target_features), min_len, replace=False)
    
    source_features = source_features[source_idx]
    target_features = target_features[target_idx]
    
    X = np.concatenate([source_features, target_features], axis=0)
    y = np.concatenate([np.zeros(min_len), np.ones(min_len)], axis=0)
    
    # 70/30 split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)
    
    # Logistic regression with C=1
    clf = LogisticRegression(C=1.0, max_iter=1000, class_weight='balanced', random_state=seed)
    clf.fit(X_train, y_train)
    
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    
    return acc * 100.0 # Return as percentage
