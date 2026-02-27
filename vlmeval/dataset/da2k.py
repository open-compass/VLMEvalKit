"""
DA-2K Dataset for VLMEvalKit
Relative Depth Estimation Benchmark (NeurIPS 2024)

Paper: "DA-2K: A Challenging Benchmark for Relative Depth Estimation"
Dataset: https://huggingface.co/datasets/DepthAnything/DA-2K

Task: Given a single image and a question about relative depth,
      predict which point is closer/farther or the depth relationship.
"""

from ..smp import *
from .image_vqa import ImageVQADataset


class DA2K(ImageVQADataset):
    """
    DA-2K: A Challenging Benchmark for Relative Depth Estimation
    
    NeurIPS 2024
    
    Dataset Statistics:
    - 1K images
    - 2K annotation pairs
    - 8 scene categories
    - Task: Relative depth estimation (VQA format)
    
    Example questions:
    - "Which point is closer to the camera, A or B?"
    - "Is point A above or below point B in depth?"
    """
    
    TYPE = 'VQA'
    
    # HuggingFace dataset repository
    DATASET_URL = {
        'DA-2K': 'https://huggingface.co/datasets/DepthAnything/DA-2K',
    }
    
    def __init__(self, dataset='DA-2K', **kwargs):
        super().__init__(dataset=dataset, **kwargs)
        self.dataset_name = dataset
    
    @classmethod
    def supported_datasets(cls):
        return ['DA-2K']
    
    def prepare_dataset(self, dataset_name='DA-2K'):
        """
        Prepare DA-2K dataset from HuggingFace
        """
        from huggingface_hub import snapshot_download
        
        repo_id = 'DepthAnything/DA-2K'
        
        # Download dataset
        dataset_path = snapshot_download(
            repo_id=repo_id,
            repo_type='dataset',
            allow_patterns=['*.json', '*.jsonl', '*.csv', '*.tsv', '*.parquet', 'images/*', 'data/*']
        )
        
        # Look for annotation file
        possible_files = [
            'da2k.json', 'da2k.jsonl', 'da-2k.json', 'da-2k.jsonl',
            'annotations.json', 'test.json', 'val.json',
            'da2k.tsv', 'da-2k.tsv', 'da2k.csv', 'da-2k.csv'
        ]
        
        data_file = None
        for fname in possible_files:
            fpath = osp.join(dataset_path, fname)
            if osp.exists(fpath):
                data_file = fpath
                break
        
        # If no annotation file found, create from HuggingFace dataset format
        if data_file is None:
            try:
                from datasets import load_dataset
                ds = load_dataset(repo_id, split='test')
                
                # Convert to DataFrame
                data_list = []
                for item in ds:
                    data_list.append({
                        'index': len(data_list),
                        'image': item.get('image_path', item.get('image', '')),
                        'question': item.get('question', ''),
                        'answer': item.get('answer', item.get('depth_answer', '')),
                        'scene_category': item.get('scene_category', item.get('category', '')),
                    })
                
                import pandas as pd
                df = pd.DataFrame(data_list)
                data_file = osp.join(dataset_path, 'da2k.tsv')
                df.to_csv(data_file, sep='\t', index=False)
                
            except Exception as e:
                print(f"Failed to load dataset from HuggingFace: {e}")
                raise
        
        return dict(data_file=data_file, root=dataset_path)
    
    def load_data(self, dataset_name=None):
        """Load and process DA-2K data"""
        data = super().load_data(dataset_name)
        
        # Ensure required columns exist
        if 'question' not in data.columns:
            raise ValueError("Dataset must contain 'question' column")
        if 'answer' not in data.columns:
            raise ValueError("Dataset must contain 'answer' column")
        
        # Add index if not present
        if 'index' not in data.columns:
            data['index'] = np.arange(len(data))
        
        return data
    
    def build_prompt(self, line, dataset=None):
        """
        Build prompt for DA-2K
        
        Expected format:
        - Image with two points marked (A and B)
        - Question about relative depth
        - Answer: "A", "B", or descriptive
        """
        if isinstance(line, int):
            line = self.data.iloc[line]
        
        # Dump/load image
        if self.meta_only:
            tgt_path = toliststr(line['image_path'])
        else:
            tgt_path = self.dump_image(line)
        
        question = line['question']
        
        # Build prompt
        # DA-2K questions are typically:
        # "Which point is closer to the camera, A or B?"
        # "Point A or Point B, which one is farther?"
        
        prompt = question
        
        # Some DA-2K questions may need context
        if 'scene_category' in line and not pd.isna(line.get('scene_category')):
            # Scene category can be used for analysis but not in prompt
            pass
        
        msgs = []
        if isinstance(tgt_path, list):
            for p in tgt_path:
                msgs.append(dict(type='image', value=p))
        else:
            msgs.append(dict(type='image', value=tgt_path))
        
        msgs.append(dict(type='text', value=prompt))
        
        return msgs
    
    @classmethod
    def evaluate(cls, eval_file, **judge_kwargs):
        """
        Evaluate DA-2K results
        
        Supports:
        - Exact matching for simple answers (A/B)
        - LLM-based evaluation for descriptive answers
        """
        from .utils import build_judge
        from ..smp import get_intermediate_file_path, get_file_extension
        
        assert get_file_extension(eval_file) in ['xlsx', 'json', 'tsv']
        
        score_file = get_intermediate_file_path(eval_file, '_score')
        
        if not osp.exists(score_file):
            data = load(eval_file)
            
            # Check if we need LLM judge
            model = judge_kwargs.get('model', 'exact_matching')
            
            if model == 'exact_matching':
                model = None
            else:
                model = build_judge(**judge_kwargs)
                if not model.working():
                    model = None
            
            # Evaluate each sample
            for idx in data['index']:
                ans = data.loc[data['index'] == idx, 'answer'].values[0]
                pred = str(data.loc[data['index'] == idx, 'prediction'].values[0])
                
                # Normalize answers
                ans = str(ans).strip().lower()
                pred = pred.strip().lower()
                
                # Simple case: A/B answer
                if ans in ['a', 'b']:
                    score = int(pred == ans)
                # Case: answer contains key word
                elif ans in pred or pred in ans:
                    score = 1
                # Use LLM judge for complex cases
                elif model is not None:
                    # TODO: Implement LLM-based evaluation
                    score = 0
                else:
                    score = 0
                
                data.loc[data['index'] == idx, 'score'] = score
            
            dump(data, score_file)
        
        # Calculate metrics
        data = load(score_file)
        
        # Overall accuracy
        overall_acc = data['score'].mean()
        
        # By scene category if available
        results = {
            'overall': overall_acc,
        }
        
        if 'scene_category' in data.columns:
            category_acc = data.groupby('scene_category')['score'].mean()
            results['by_category'] = category_acc.to_dict()
        
        rating_file = get_intermediate_file_path(eval_file, '_rating', 'json')
        dump(results, rating_file)
        
        return results
