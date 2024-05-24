# 微调qwen1.5B实战

利用微调工具LLaMA-Factory和大模型评测工具LLMuse，参考文章https://blog.csdn.net/qq_45156060/article/details/136741929

1、数据准备，参考项目https://github.com/liukangjia666/LLM_fine_tuning/tree/main/LLM_data，至少需要两个数据，train.json、dataset_info.json，将数据放在data（LLaMA-Factory）文件夹下

2、环境搭建

LLama-Factory

>git clone https://github.com/hiyouga/LLaMA-Factory.git
>conda create -n llama_factory python=3.10
>conda activate llama_factory
>cd LLaMA-Factory
>pip install -r requirements.txt
>pip install modelscope -Uexport USE_MODELSCOPE_HUB=1
>
>注意，目前环境目录已经变化，可以使用本项目代码

llmuses（https://github.com/modelscope/eval-scope）:

>conda create -n eval-scope python=3.10
>conda activate eval-scope
>
>pip install llmuses
>
>例子：python -m llmuses.run --model ZhipuAI/chatglm3-6b --template-type chatglm3 --datasets arc --limit 100

3、训练

>CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \
>    --stage sft \
>    --do_train \
>    --model_name_or_path qwen/Qwen1.5-1.8B-Chat \
>    --dataset train \
>    --template qwen \
>    --finetuning_type lora \
>    --lora_target q_proj,v_proj \
>    --output_dir output\
>    --overwrite_cache \
>    --per_device_train_batch_size 2 \
>    --gradient_accumulation_steps 32 \
>    --lr_scheduler_type cosine \
>    --logging_steps 10 \
>    --save_steps 1000 \
>    --learning_rate 5e-5 \
>    --num_train_epochs 3.0 \
>    --plot_loss \
>    --fp16
>
>

4、模型合并

>CUDA_VISIBLE_DEVICES=0 python src/export_model.py \
>    --model_name_or_path qwen/Qwen1.5-1.8B-Chat\
>    --adapter_name_or_path output \
>    --template qwen \
>    --finetuning_type lora \
>    --export_dir Qwen1.5-1.8B-Chat_fine \
>    --export_size 2 \
>    --export_legacy_format False

5、结果，见result目录

微调遇到的问题汇总

1、一开始按照文章的指导，使用GPU算力白嫖中资源，但是他的镜像本来已经安装的cuda版本和conda自己创建的环境老是不能兼容，各种奇奇怪怪的问题，最后换成了收费的AutoDL，问题解决

2、安装依赖的出问题时候不要慌，换个源试试，问题可能就解决了

3、在第四步，模型合并，一直报 cannot import name 'MixtralBLockSparseTop2MLP' from 'transformers.models.mixtral.modeling_mixtral'

修改import顺序后解决，不知道啥原因

4、两个数据结果在result中，微调后效果更差一些，不知道啥原因

5、qwen1.5B的模型微调中占用显存大概5G左右

