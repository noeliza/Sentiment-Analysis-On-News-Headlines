from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import re
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize, pos_tag
from nltk.corpus import stopwords
# import nltk
# nltk.download('stopwords')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('wordnet')
# nltk.download('omw-1.4')



def view_wordcloud(strings, title = None):
    stopwords = set(STOPWORDS)

    wordcloud = WordCloud(width=1600, height=800,
                    background_color ='white',
                    stopwords = stopwords).generate(strings)

    plt.figure(figsize = (20, 10), facecolor = None)
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.tight_layout(pad = 0)
    if title != None:
        plt.title(title, fontsize=50, fontweight = 'bold')
    plt.show()

    
    
class preprocess_text:
    
    def __init__(self, input_data):
        self.data = input_data
     
    def standardize_text(self, headline):
        # for replacing exact workd
        headline = headline.replace("Gov't", 'Government')
        headline = headline.replace("Dev't", 'Development')
        headline = headline.replace("Addt'l", 'Additional')  
        headline = headline.replace("Add'l", 'Additional')
        headline = headline.replace("BSP", 'Central Bank of the Philippines')
        headline = headline.replace("PH", 'Philippines')
        headline = headline.replace("-yr", '-Year')
        headline = headline.replace("yrs", 'Years')

        return headline

    def clean_text(self, headline):
        # remove excess/trailing whitespace and punctuations
        headline = re.sub(r"[\.]", '', headline)
        headline = re.sub(r"[\'’‘]", ' ', headline)
        headline = re.sub(r'([^a-zA-Z ]+?)', ' ', headline)

        # remove string with one character only except whitespace
        tokens = word_tokenize(headline)
        tokens = [token for token in tokens if len(token) >= 2]

        new_headline = ' '.join(tokens)
        return new_headline
    
    def lemmatize(self, headline):
        stopwords_nltk = stopwords.words('english') 
        for index, x in enumerate(stopwords_nltk): 
            stopwords_nltk[index] = re.sub(r"['’‘]", "", x)

        merge_stopwords = STOPWORDS.union(stopwords_nltk)

        wordnet_tag ={'NN':'n','JJ':'a','VB':'v','RB':'r'}
        lemma = []
        tagging = pos_tag(word_tokenize(headline))
        for token in tagging:
            try: lemma.append(WordNetLemmatizer().lemmatize(token[0], wordnet_tag[token[1][:2]]))
            except: lemma.append(WordNetLemmatizer().lemmatize(token[0]))
            lemmatize_headline = ' '.join(lemma)
        return(lemmatize_headline)
    
    def run(self):
        print(f' Lowercase text...')
        self.data['lowercase']= self.data.headlines.str.lower()
        print(f' Standardize text...')
        self.data['stdz'] = self.data.lowercase.apply(lambda item: self.standardize_text(item))
        print(f' Remove excess/trailing white space, and punctations...')
        print(f' Remove string with one character only except whitespace...')
        self.data['clean'] = self.data.stdz.apply(lambda item: self.clean_text(item))
        print(f' Lemmatize text...')
        self.data['lemmatize'] = self.data.clean.apply(lambda item: self.lemmatize(item))
          
        return self.data[['headlines', 'lemmatize', 'label']]

# create roberta encoder function for model explainability    
class transformer:
    def __init__(self, input):
        self.transformer = input
        
    def roberta_encode(self, item):
        return self.transformer.encode(item, convert_to_tensor = True, show_progress_bar = False).detach().cpu().numpy()