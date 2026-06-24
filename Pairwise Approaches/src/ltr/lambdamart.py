"""
ltr/lambdamart.py
-----------------
LambdaMART implementation using scikit-learn DecisionTreeRegressor as the base
learner. Features an O(n^2) path-following optimal line search to find the exact
step size (alpha) that maximizes NDCG at each iteration, natively handling degeneracies
via score jittering.
"""

import numpy as np
import torch
from sklearn.tree import DecisionTreeRegressor
from typing import List, Tuple, Dict
from tqdm import tqdm

from .losses import lambda_gradients_and_hessians
from .metrics import mean_ndcg, ndcg_at_k


class LambdaMART:
    """
    LambdaMART with Optimal Line Search.
    
    Instead of using a fixed learning rate (shrinkage), this implementation
    analytically calculates the optimal step size (alpha) for each tree by
    performing an exact path-following line search to maximize NDCG.
    """
    def __init__(
        self,
        n_estimators: int = 1000,
        learning_rate: float = 0.1,  # Fallback if line search fails or is disabled
        max_depth: int = 6,
        min_samples_leaf: int = 20,
        k: int = 10,
        patience: int = 10,
        use_optimal_line_search: bool = True
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.k = k
        self.patience = patience
        self.use_optimal_line_search = use_optimal_line_search
        
        self.trees = []
        self.alphas = []
        
    def _find_crossing_points(
        self, scores: np.ndarray, preds: np.ndarray, labels: np.ndarray
    ) -> List[Tuple[float, int, int]]:
        """
        Finds all crossing points alpha for a single query where:
        S_i + alpha * P_i = S_j + alpha * P_j
        """
        N = len(labels)
        crossings = []
        
        # Add jitter to predictions to prevent degeneracies
        # A microscopic amount of noise ensures no two lines are perfectly parallel
        # or cross at the exact same alpha, allowing sequential processing.
        preds_jittered = preds + np.random.normal(0, 1e-9, size=preds.shape)
        
        for i in range(N):
            for j in range(i + 1, N):
                if labels[i] != labels[j]:
                    delta_P = preds_jittered[i] - preds_jittered[j]
                    if delta_P != 0:
                        alpha = (scores[j] - scores[i]) / delta_P
                        if alpha > 0:
                            crossings.append((alpha, i, j))
        return crossings

    def _optimal_line_search(
        self, all_qids, current_scores, tree_preds, labels_dict
    ) -> float:
        """
        Finds the global optimal alpha that maximizes NDCG across all queries.
        """
        # 1. Collect all crossings across all queries
        all_crossings = []
        for qid in all_qids:
            scores_q = current_scores[qid]
            preds_q = tree_preds[qid]
            labels_q = labels_dict[qid]
            
            crossings = self._find_crossing_points(scores_q, preds_q, labels_q)
            for alpha, i, j in crossings:
                # Limit alpha to a reasonable range as suggested by the paper [0.1, 100]
                # We use [1e-4, 100] to be a bit more flexible
                if 1e-4 <= alpha <= 100.0:
                    all_crossings.append((alpha, qid, i, j))
                    
        # Sort crossings by alpha
        all_crossings.sort(key=lambda x: x[0])
        
        # 2. Track ranks and base NDCG
        rank_indices = {}
        for qid in all_qids:
            # Add jitter to base scores to ensure strict initial ordering without ties
            jittered_scores = current_scores[qid] + np.random.normal(0, 1e-9, size=current_scores[qid].shape)
            rank_indices[qid] = np.argsort(jittered_scores)[::-1]
            
        # Calculate base NDCG
        current_ndcg_per_query = {}
        for qid in all_qids:
            labels_q = labels_dict[qid]
            scores_q = current_scores[qid]
            current_ndcg_per_query[qid] = ndcg_at_k(labels_q, scores_q, self.k)
            
        base_ndcg = np.mean(list(current_ndcg_per_query.values())) if current_ndcg_per_query else 0.0
        
        max_ndcg = base_ndcg
        best_alpha = 0.0
        
        # Ideal DCG per query for fast updates
        idcg_dict = {}
        for qid in all_qids:
            labels_q = labels_dict[qid]
            sorted_labels = np.sort(labels_q)[::-1]
            def dcg(rel, k):
                r = np.asarray(rel)[:k]
                if r.size == 0: return 0.0
                denoms = np.log2(np.arange(2, r.size + 2))
                return float(np.sum((2.0 ** r - 1.0) / denoms))
            idcg_dict[qid] = dcg(sorted_labels, self.k)
        
        running_ndcg_sum = base_ndcg * len(all_qids)
        
        # Precompute discounts
        max_N = max([len(labels_dict[q]) for q in all_qids]) if all_qids else 0
        discounts = np.zeros(max_N)
        for r in range(min(max_N, self.k)):
            discounts[r] = 1.0 / np.log2(r + 2.0)
            
        # Inverse mapping: pos_of[qid][doc_index] = rank_position
        pos_of = {}
        for qid in all_qids:
            pos_of[qid] = np.zeros(len(labels_dict[qid]), dtype=int)
            for pos, doc_idx in enumerate(rank_indices[qid]):
                pos_of[qid][doc_idx] = pos
                
        # 3. Traverse path
        for alpha, qid, i, j in all_crossings:
            idcg = idcg_dict[qid]
            if idcg == 0:
                continue
                
            labels_q = labels_dict[qid]
            pos_i = pos_of[qid][i]
            pos_j = pos_of[qid][j]
            
            # Compute old DCG contribution for these two docs
            old_dcg = 0.0
            if pos_i < self.k:
                old_dcg += (2.0 ** labels_q[i] - 1.0) * discounts[pos_i]
            if pos_j < self.k:
                old_dcg += (2.0 ** labels_q[j] - 1.0) * discounts[pos_j]
                
            # Swap
            rank_indices[qid][pos_i], rank_indices[qid][pos_j] = rank_indices[qid][pos_j], rank_indices[qid][pos_i]
            pos_of[qid][i] = pos_j
            pos_of[qid][j] = pos_i
            
            # Compute new DCG contribution
            new_pos_i = pos_j
            new_pos_j = pos_i
            new_dcg = 0.0
            if new_pos_i < self.k:
                new_dcg += (2.0 ** labels_q[i] - 1.0) * discounts[new_pos_i]
            if new_pos_j < self.k:
                new_dcg += (2.0 ** labels_q[j] - 1.0) * discounts[new_pos_j]
                
            delta_ndcg = (new_dcg - old_dcg) / idcg
            
            running_ndcg_sum += delta_ndcg
            current_mean_ndcg = running_ndcg_sum / len(all_qids)
            
            if current_mean_ndcg > max_ndcg:
                max_ndcg = current_mean_ndcg
                best_alpha = alpha
                
        # If no alpha improved NDCG, fallback to the learning rate
        if best_alpha == 0.0:
            best_alpha = self.learning_rate
            
        return best_alpha

    def fit(self, train_loader, val_loader, device='cpu', verbose=True):
        best_val_ndcg = -1.0
        epochs_no_improve = 0
        best_iteration = 0
        
        # Extract training data
        all_train_qids = []
        train_feats_dict = {}
        train_labels_dict = {}
        train_scores_dict = {}
        
        for batch_qids, batch_feats, batch_labels in train_loader:
            for qid, feats, labels in zip(batch_qids, batch_feats, batch_labels):
                qkey = str(qid.item() if isinstance(qid, torch.Tensor) else qid)
                all_train_qids.append(qkey)
                train_feats_dict[qkey] = feats.numpy() if isinstance(feats, torch.Tensor) else np.asarray(feats)
                train_labels_dict[qkey] = labels.numpy() if isinstance(labels, torch.Tensor) else np.asarray(labels)
                train_scores_dict[qkey] = np.zeros(len(labels))
                
        val_ndcg_history = []
                
        for iteration in range(self.n_estimators):
            
            # 1. Compute lambdas and hessians
            all_feats = []
            all_lambdas = []
            all_hessians = []
            
            for qid in all_train_qids:
                scores = torch.tensor(train_scores_dict[qid], dtype=torch.float32).unsqueeze(1)
                labels = torch.tensor(train_labels_dict[qid], dtype=torch.float32)
                
                lambdas, hessians = lambda_gradients_and_hessians(scores, labels, k=self.k)
                
                all_feats.append(train_feats_dict[qid])
                all_lambdas.append(lambdas.numpy().squeeze())
                all_hessians.append(hessians.numpy().squeeze())
                
            X = np.vstack(all_feats)
            lambdas = np.concatenate(all_lambdas)
            hessians = np.concatenate(all_hessians)
            
            # Avoid division by zero
            w = np.clip(hessians, 1e-10, None)
            
            # Pseudo-residuals for Newton step: -grad / hess
            # Since our lambda_gradients compute the gradient (negative means score should go up),
            # the step is -lambda/w. So target is -lambda/w.
            target = -lambdas / w
            
            # 2. Fit tree
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                random_state=42 + iteration
            )
            # Fitting with sample_weight=w makes the leaf values exactly sum(w_i * target_i) / sum(w_i)
            # which equals sum(-lambda_i) / sum(w_i), matching the exact Newton step formula.
            tree.fit(X, target, sample_weight=w)
            
            # Get predictions
            tree_preds_dict = {}
            for qid in all_train_qids:
                tree_preds_dict[qid] = tree.predict(train_feats_dict[qid])
                
            # 3. Find optimal alpha
            if self.use_optimal_line_search:
                alpha = self._optimal_line_search(
                    all_train_qids, train_scores_dict, tree_preds_dict, train_labels_dict
                )
            else:
                alpha = self.learning_rate
                
            self.trees.append(tree)
            self.alphas.append(alpha)
            
            # 4. Update scores
            for qid in all_train_qids:
                train_scores_dict[qid] += alpha * tree_preds_dict[qid]
                
            # 5. Validation and Early Stopping
            val_ndcg = mean_ndcg(self, val_loader, k_list=(self.k,), device=device)[self.k]
            val_ndcg_history.append(val_ndcg)
            
            improved = val_ndcg > best_val_ndcg
            if improved:
                best_val_ndcg = val_ndcg
                best_iteration = iteration
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                
            if verbose:
                marker = "  <- best" if improved else ""
                print(f"Tree {iteration+1:03d} | alpha: {alpha:.4f} | Val NDCG@{self.k}: {val_ndcg:.4f}{marker}")
                
            if self.patience > 0 and epochs_no_improve >= self.patience:
                if verbose:
                    print(f"Early stopping at tree {iteration+1} (no improvement for {self.patience} trees).")
                break
                
        # Truncate trees to best iteration
        if best_iteration + 1 < len(self.trees):
            self.trees = self.trees[:best_iteration + 1]
            self.alphas = self.alphas[:best_iteration + 1]
            
        return self, val_ndcg_history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Sum of all trees scaled by their alpha."""
        if not self.trees:
            return np.zeros(X.shape[0])
            
        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy()
            
        preds = np.zeros(X.shape[0])
        for tree, alpha in zip(self.trees, self.alphas):
            preds += alpha * tree.predict(X)
            
        return preds

    def __call__(self, X: torch.Tensor) -> torch.Tensor:
        """PyTorch Module compatibility for mean_ndcg evaluation."""
        preds = self.predict(X)
        return torch.tensor(preds, dtype=torch.float32, device=X.device).unsqueeze(1)
        
    def eval(self):
        """Dummy method for PyTorch compatibility."""
        pass
