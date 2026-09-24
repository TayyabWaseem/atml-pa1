import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def compute_source_domain_separability(backbone, source_val_loaders, device, seed=6304):
    """
    Computes domain separability score using a balanced logistic regression classifier
    to predict Photo(0), Art(1), Cartoon(2).
    """
    backbone.eval()
    
    features_by_domain = {"photo": [], "art_painting": [], "cartoon": []}
    domain_to_label = {"photo": 0, "art_painting": 1, "cartoon": 2}
    
    with torch.no_grad():
        for domain, loader in source_val_loaders.items():
            if domain not in features_by_domain: continue
            for x, _ in loader:
                x = x.to(device)
                feats = backbone(x)
                features_by_domain[domain].extend(feats.cpu().numpy())
                
    # Balance features
    min_len = min(len(feats) for feats in features_by_domain.values())
    np.random.seed(seed)
    
    X_list = []
    y_list = []
    
    for domain, feats in features_by_domain.items():
        feats = np.array(feats)
        idx = np.random.choice(len(feats), min_len, replace=False)
        X_list.append(feats[idx])
        y_list.append(np.full(min_len, domain_to_label[domain]))
        
    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    
    # 70/30 split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)
    
    # Logistic regression with C=1, multinomial automatically handled
    clf = LogisticRegression(C=1.0, max_iter=1000, multi_class='multinomial', random_state=seed)
    clf.fit(X_train, y_train)
    
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    
    return acc * 100.0
