from modelscope import snapshot_download
from scripts import config

# modelscope download --model 'Qwen/Qwen2-7b' --local_dir './'

print(snapshot_download(model_id=config.model_name, cache_dir='./qwen/'))

