docs=20
SUMMARY="/tmp/pycharm_project_631/result_abstract/40_60_biobert_simi_wa_d100_j10_trec_fields.${docs}.summary"
printf "run\tqrels\tmeasure\ttopic\tscore\n" > $SUMMARY
RUN_FILE_PATH="/tmp/pycharm_project_631/result_abstract/40_60_biobert_simi_wa_d100_j10_trec_fields.csv"

QRELS="/tmp/pycharm_project_631/qrels"
trec_eval='/home/ubuntu/rupadhyay/Trec_eval_extension/'
cd $trec_eval
#compatibility="/home/ricky/Documents/PhDproject/Project_folder/Compatibility/compatibility.py"

$trec_eval/trec_eval -q -c -M ${docs} -m cam_map -R qrels_twoaspects $QRELS/misinfo-qrels.2aspects.useful-credible $RUN_FILE_PATH | gawk '{print "'$RUN_NAME'" "\t" "2aspects.useful-credible" "\t" $1 "\t" $2 "\t" $3}' >> $SUMMARY

$trec_eval/trec_eval -q -c -M ${docs} -m cam -R qrels_twoaspects $QRELS/misinfo-qrels.2aspects.useful-credible $RUN_FILE_PATH | gawk '{print "'$RUN_NAME'" "\t" "2aspects.useful-credible" "\t" $1 "\t" $2 "\t" $3}' >> $SUMMARY

#$trec_eval -q -c -M ${docs} -m nwcs -R qrels_twoaspects $QRELS/misinfo-qrels.2aspects.useful-credible $RUN_FILE_PATH | gawk '{print "'$RUN_NAME'" "\t" "2aspects.useful-credible" "\t" $1 "\t" $2 "\t" $3}' >> $SUMMARY

#$trec_eval -q -c -M 10 -m nlre -R qrels_twoaspects $QRELS/misinfo-qrels.2aspects.useful-credible $RUN_FILE_PATH | gawk '{print "'$RUN_NAME'" "\t" "2aspects.useful-credible" "\t" $1 "\t" $2 "\t" $3}' >> $SUMMARY

#python3 $compatibility $QRELS/misinfo-qrels-graded.harmful-only $RUN_FILE_PATH | gawk '{print "'$RUN_NAME'" "\t" "graded.harmful-only" "\t" $1 "\t" $2 "\t" $3}' >> $SUMMARY
#python3 $compatibility $QRELS/misinfo-qrels-graded.helpful-only $RUN_FILE_PATH | gawk '{print "'$RUN_NAME'" "\t" "graded.helpful-only" "\t" $1 "\t" $2 "\t" $3}' >> $SUMMARY