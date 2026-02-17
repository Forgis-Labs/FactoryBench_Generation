import numpy as np
import pandas as pd
from scipy.linalg import eigvalsh
from sklearn.metrics.pairwise import cosine_similarity, rbf_kernel
import warnings
warnings.filterwarnings('ignore')

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("Warning: sentence-transformers not installed. Install with: pip install sentence-transformers")

class VENDICalculator:
    """
    Calculate VENDI diversity score for text datasets
    """
    
    def __init__(self, model_name='all-MiniLM-L6-v2', kernel_type='cosine'):
        """
        Initialize VENDI calculator
        
        Args:
            model_name: Sentence transformer model name
            kernel_type: 'cosine' or 'rbf' for similarity computation
        """
        self.kernel_type = kernel_type
        
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.encoder = SentenceTransformer(model_name)
            except Exception as e:
                print(f"Error loading SentenceTransformer: {e}")
                self.encoder = None
        else:
            self.encoder = None
    
    def compute_embeddings(self, texts):
        """
        Compute embeddings for texts
        
        Args:
            texts: List of strings
        
        Returns:
            np.array of embeddings
        """
        if self.encoder is None:
            raise RuntimeError("sentence-transformers not installed or model failed to load")
        
        embeddings = self.encoder.encode(
            texts, 
            show_progress_bar=False, # cleaner output
            convert_to_numpy=True
        )
        
        return embeddings
    
    def compute_kernel_matrix(self, embeddings):
        """
        Compute similarity kernel matrix
        
        Args:
            embeddings: np.array of embeddings (N x D)
        
        Returns:
            Kernel matrix K (N x N)
        """
        if self.kernel_type == 'cosine':
            # Cosine similarity
            K = cosine_similarity(embeddings)
        elif self.kernel_type == 'rbf':
            # RBF (Gaussian) kernel
            K = rbf_kernel(embeddings)
        else:
            raise ValueError(f"Unknown kernel type: {self.kernel_type}")
        
        # Normalize
        N = len(embeddings)
        K = K / N
        
        return K
    
    def von_neumann_entropy(self, K, epsilon=1e-10):
        """
        Compute von Neumann entropy of kernel matrix
        
        H(K) = -sum(λ_i * log(λ_i))
        where λ_i are eigenvalues of K
        
        Args:
            K: Kernel matrix (N x N)
            epsilon: Threshold for numerical stability
        
        Returns:
            Entropy value
        """
        # Compute eigenvalues
        eigenvalues = eigvalsh(K)
        
        # Filter small eigenvalues for numerical stability
        eigenvalues = eigenvalues[eigenvalues > epsilon]
        
        # Normalize to sum to 1 (probability distribution)
        eigenvalues = eigenvalues / eigenvalues.sum()
        
        # Compute entropy: H = -sum(p * log(p))
        entropy = -np.sum(eigenvalues * np.log(eigenvalues + epsilon))
        
        return entropy
    
    def vendi_score(self, texts=None, embeddings=None, K=None):
        """
        Compute VENDI score
        
        VS = exp(H(K))
        
        Can provide either:
        - texts: compute embeddings and kernel
        - embeddings: compute kernel
        - K: use directly
        
        Args:
            texts: List of strings (optional)
            embeddings: Pre-computed embeddings (optional)
            K: Pre-computed kernel matrix (optional)
        
        Returns:
            VENDI score (float)
        """
        if K is None:
            if embeddings is None:
                if texts is None:
                    raise ValueError("Must provide texts, embeddings, or K")
                embeddings = self.compute_embeddings(texts)
            
            K = self.compute_kernel_matrix(embeddings)
        
        H = self.von_neumann_entropy(K)
        VS = np.exp(H)
        
        return VS
    
    def effective_diversity(self, vendi_score, n_samples):
        """
        Compute effective diversity as fraction of dataset
        
        Returns:
            Normalized diversity score [0, 1]
        """
        return vendi_score / n_samples
    
    def analyze_diversity(self, questions_df, group_by='level', text_column='question'):
        """
        Analyze diversity across groups
        
        Args:
            questions_df: DataFrame with questions
            group_by: Column to group by (e.g., 'level', 'type')
            text_column: Column containing question text
        
        Returns:
            dict with diversity metrics per group
        """
        results = {}
        
        if group_by not in questions_df.columns:
             # Fallback: treat whole DF as one group if column missing
             # or just return empty
             # For robustness let's create a dummy group
             questions_df['all'] = 'all'
             group_by = 'all'

        for group_name, group_df in questions_df.groupby(group_by):
            questions = group_df[text_column].tolist()
            
            if len(questions) < 2:
                results[group_name] = {
                    'n_questions': len(questions),
                    'vendi_score': np.nan,
                    'effective_diversity': np.nan,
                    'note': 'Too few samples'
                }
                continue
            
            try:
                vs = self.vendi_score(texts=questions)
                eff_div = self.effective_diversity(vs, len(questions))
                
                results[group_name] = {
                    'n_questions': len(questions),
                    'vendi_score': vs,
                    'effective_diversity': eff_div,
                    'diversity_percentage': eff_div * 100
                }
                
            except Exception as e:
                results[group_name] = {
                    'n_questions': len(questions),
                    'vendi_score': np.nan,
                    'effective_diversity': np.nan,
                    'error': str(e)
                }
        
        return results
    
    def print_diversity_report(self, diversity_results):
        """
        Print formatted diversity analysis report
        """
        print("\n" + "="*70)
        print("VENDI DIVERSITY ANALYSIS REPORT")
        print("="*70 + "\n")
        
        for group, metrics in diversity_results.items():
            print(f"Group: {group}")
            print(f"  Number of questions: {metrics['n_questions']}")
            
            if pd.isna(metrics.get('vendi_score')):
                print(f"  Status: {metrics.get('note', 'Error')}")
            else:
                print(f"  VENDI Score: {metrics['vendi_score']:.2f}")
                print(f"  Effective Diversity: {metrics['effective_diversity']:.3f} ({metrics['diversity_percentage']:.1f}%)")
                
                # Interpret score
                if metrics['effective_diversity'] > 0.5:
                    status = "✅ Good diversity"
                elif metrics['effective_diversity'] > 0.3:
                    status = "⚠️ Moderate diversity - could improve"
                else:
                    status = "❌ Low diversity - needs improvement"
                
                print(f"  Assessment: {status}")
            
            print()
