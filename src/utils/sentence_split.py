import spacy
spacy_nlp = spacy.load("xx_ent_wiki_sm")
spacy_nlp.add_pipe('sentencizer')


def sentence_splitter(text):
	doc = spacy_nlp(text)
	sentences = []
	return [sent.text.strip() for sent in doc.sents]