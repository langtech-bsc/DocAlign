import spacy
spacy_nlp = spacy.load("xx_ent_wiki_sm")
spacy_nlp.add_pipe('sentencizer')


def sentence_splitter(text):
    sentences = []
    for sentence in text:
        sentences.extend(sentence.text.strip() for sentence in spacy_nlp(sentence).sents)

    return sentences