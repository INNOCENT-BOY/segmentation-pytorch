CUDA_VISIBLE_DEVICES=2 \
python ./deploy.py \
--model_path ~/work/lumber/working_directory/lumber/models/chongyan \
--input_path ~/work/lumber/working_directory/lumber/inputs \
--output_path ~/work/lumber/working_directory/lumber/outputs \
--target_file_patterns zhengguang*.bmp \
--tta