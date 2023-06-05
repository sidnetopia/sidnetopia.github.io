import nltk
from nltk.tag.stanford import StanfordPOSTagger as POS_Tag
from nltk.tokenize import word_tokenize

if __name__ == '__main__':
    nltk.download('punkt')
    tagger_dir = '/home/sidnetopia/Documents/nlp/assignments/part_of_speech_tagging/pos/tagger'
    tagger_model = 'filipino-left5words-owlqn2-distsim-pref6-inf2.tagger'
    tagger_jar = 'stanford-postagger.jar'

    pos_tagger = POS_Tag(f"{tagger_dir}/{tagger_model}", f"{tagger_dir}/{tagger_jar}")

    text_list = [
        "Pumunta sa palengke ang Hapon.",
        "Pumunta sa palengke ang Hapon kaninang hapon.",
        "Hapong-hapo ang Hapon na pumunta sa palengke kaninang hapon."
    ]
    for text in text_list:
        print(f"Sentence: {text}")
 
        words = word_tokenize(text)

        tagged_words = pos_tagger.tag(words)
        print(tagged_words)

        for word,word_class in tagged_words:
            print(word + "_" + word_class)

        print(" ")