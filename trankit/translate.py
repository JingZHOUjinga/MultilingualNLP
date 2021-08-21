#用transformers中的MarianMTModel实现英文到中文的翻译
from transformers import MarianTokenizer, MarianMTModel

def trans_lang(src,trg,sent):
    #src = 'en'
    #trg = 'zh'
    #sample_text = "this is a sentence in english that we want to translate to chinese"

    """
    src:源语言
    trg:目标语言
    sent:输入语句
    result:翻译语句列表，类型list
    """
    
    model_name = f'Helsinki-NLP/opus-mt-{src}-{trg}'
    model = MarianMTModel.from_pretrained(model_name)
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    batch = tokenizer([sent], return_tensors="pt")
    gen = model.generate(**batch)
    result = tokenizer.batch_decode(gen, skip_special_tokens=True)
    print(result)
    return result

if __name__ == '__main__':
    src = 'de'
    trg = 'ZH'
    sample_text = "Der chinesische Technologieriese Tencent hat angekündigt, genau 50 Milliarden Yuan (6,6 Mrd. Euro) investieren zu wollen, um die Initiative für gemeinsamen Wohlstand der chinesischen Regierung zu fördern. Dies teilte das Unternehmen am Mittwoch über seinen offiziellen WeChat-Account mit."
    result = trans_lang(src,trg,sample_text)
    print(result)