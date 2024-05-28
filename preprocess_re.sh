#for data_dir in "en" "zh" "de" "jp" "en_zh_50" "en_de_50" "en_jp_50" "zh_de_50" "zh_jp_50" "de_jp_50"; do
#  python code/process_re.py --data_dir ./data/mixre/data/$data_dir --save_dir ./data/mixre/instruct_data/$data_dir --lang $data_dir
#done

for data_dir in 'mix-30-3k' 'mix-50-3k' 'mix-70-3k' 'inter-sent-50' 'intra-sent-50' 'entity-50'; do
  python code/process_re.py --data_dir ./data/mixre/data/$data_dir --save_dir ./data/mixre/instruct_data/$data_dir --lang en
done