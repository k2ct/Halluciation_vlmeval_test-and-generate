import pandas as pd
from vlmeval.dataset.image_base import ImageBaseDataset

class MyMiniDataset(ImageBaseDataset):
    TYPE = 'VQA'
    #
    MODALITY = 'IMAGE'
    @classmethod
    def supported_datasets(cls):
        return ['my_mini_dataset']
    #    
    DATASET_URL = {}
    DATASET_MD5 = {}

    def build_prompt(self, line):
        # 这里直接使用 TSV 中的 question
        # image 字段是 base64，父类会按默认机制处理数据行
        prompt = line['question']
        image = line['image']
        return [
            dict(type='image', value=image),
            dict(type='text', value=prompt)
        ]

    @classmethod
    def evaluate(cls, eval_file, **judge_kwargs):
        # 最小冒烟测试先不做复杂评测
        # 只把预测文件读出来，返回一个简单结果，避免 all 模式报错
        df = pd.read_excel(eval_file)
        return pd.DataFrame([{
            'dataset': 'my_mini_dataset',
            'num_samples': len(df),
            'status': 'inference_finished'
        }])