CUDA_VISIBLE_DEVICES=3 \
python ./deploy.py \
--model_path ~/work/lumber/working_directory/lumber/models/qiaopi \
--input_path ~/work/lumber/working_directory/lumber/inputs \
--output_path ~/work/lumber/working_directory/lumber/outputs \
--target_file_patterns zuoguang*.bmp|youguang*.bmp \
--tta