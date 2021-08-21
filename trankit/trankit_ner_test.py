from trankit import Pipeline
import pandas as pd
from pandas.core.frame import DataFrame
#frame输出完整
pd.options.display.max_rows = None
#frame输出对齐
pd.set_option('display.unicode.ambiguous_as_wide',True)  # 将模糊字符宽度设置为2
pd.set_option('display.unicode.east_asian_width',True) # 检查东亚字符宽度属性

from translate import trans_lang


#p = Pipeline('auto')

p = Pipeline('english',gpu=True)
p.add('chinese')
p.add('french')
p.add('german')
p.set_auto(True)



#txt_file = open('bilingual.txt', 'r') 
def trankit_ner(file_path):
    txt_file = open(file_path, 'r') 
    ner_unite = []
    ner_single = []
    ner_type = []
    ner_lang = []
    sent_list = []
    trans_sent_list = []
    for line in txt_file.readlines():
        line = line.strip("\n").replace("\n", "")   
        #去除多余的空行，保证字符串有意义
        if line in ['\n','\r\n'] or line.strip() == "": 
            pass
        else:              

            #print(line)

            ner_output = p.ner(line)
            sent = ner_output['sentences']
            #sent_lang表示语种识别的语言名称
            sent_lang = ner_output['lang']
            for sent_part in sent:
                tokens_info = sent_part['tokens']
                sent_text = sent_part['text']
                for token in tokens_info:
                    #"O"表示不属于任何实体类型，是无意义的，需要过滤掉
                    if token['ner'] != "O":
                        #合并NER的开始、中间、结束
                        if 'B-' in token['ner']:
                            ner_unite.append(token['text'])
                            continue
                        elif 'I-' in token['ner']:
                            #print(token['ner'])
                            ner_unite.append(token['text'])
                            continue
                        elif 'E-' in token['ner']:
                            ner_unite.append(token['text'])
                            #print(ner_unite)
                            if sent_lang == "chinese":
                               ner_single.append(''.join(ner_unite))
                            else:
                                #当遇到非中文语言时用_将分割的NER部分相连，不然会影响阅读，印欧语系中的语言以表音构成，用空格来区分单词
                                ner_single.append('_'.join(ner_unite))
                                sent_list.append(sent_text)
                                #此处限定是英语和德语识别
                                src = "en" if sent_lang == "english" else "de"
                                trg = "zh" if sent_lang == "english" else "ZH"
                                print(sent_text)
                                trans_sent = trans_lang(src,trg,sent_text)[0]
                                print(trans_sent)
                                trans_sent_list.append(trans_sent)
                            ner_type.append(token['ner'].replace("E-", ""))
                            ner_lang.append(sent_lang)
                            ner_unite = []
                        else:
                            #其他情况表示NER部分没有被分割掉，可单独成实体
                            ner_single.append(token['text'])
                            ner_type.append(token['ner'])
                            ner_lang.append(sent_lang)

    #print(ner_single)
    print(sent_list)

    ner_dict = {"entity":ner_single,"type":ner_type,"lang":ner_lang,"sentence":sent_list,"translated":trans_sent_list}
    ner_frame = DataFrame.from_dict(ner_dict,orient='index')
    return ner_frame

if __name__ == '__main__':
    #ner_result = trankit_ner('bilingual.txt')
    ner_result = trankit_ner("de_en_news.txt")
    #ner_result = trankit_ner('GenshinImpact.txt')
    print(ner_result)
