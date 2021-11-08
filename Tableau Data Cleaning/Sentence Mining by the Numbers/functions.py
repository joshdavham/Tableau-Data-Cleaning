import nltk
import spacy
import numpy as np
import pandas as pd

nlp_fr = spacy.load('fr_core_news_sm')

def load_data(books_list):

    #contains a list of dataframes corresponding to each book
    book_dfs = []
    #contains a list of frequency list corresponding to each book
    book_freq_lists = []

    #fill the above lists
    for book in books_list:

        book_df = pd.read_csv(book + 'book_df.csv')

        freq_list_book = pd.read_csv(book + 'freq_list.csv', index_col=0)

        book_dfs.append(book_df)
        book_freq_lists.append(freq_list_book)


    #for computational and simplified purposes, we will combine each
    #books' frequency lists into a single frequency list for the series

    freq_list_series = book_freq_lists[0]
    for freq_list_book in book_freq_lists[1:]:
        freq_list_series = pd.concat([freq_list_series, freq_list_book], axis=0)

    freq_list_series = freq_list_series.drop_duplicates(subset=['word'])
    freq_list_series = freq_list_series.sort_index(ascending=True)

    return book_dfs, freq_list_series

def get_T(sentence, known_words_series):

    #counts the number of real French words in the sentence
    num_words_sentence = 0
    #counts the number of real unknown French words in the sentence
    num_unknown_words_sentence = 0

    #tokenize the sentence
    tokens = nlp_fr(sentence.lower())

    #T counts number of unknown words in the sentence
    T = 0
    #returns the unknown word
    unknown_words = set()

    for word in tokens:

        #looks for the word in the harry potter frequency list
        rslt = known_words_series.loc[known_words_series['word'] == word.lemma_, 'known']

        #does it even exist in the frequency list?
        if len(rslt) != 0:

            num_words_sentence = num_words_sentence + 1

            #the word is not a known word
            if int(rslt) == 0:

                num_unknown_words_sentence = num_unknown_words_sentence + 1

                unknown_words.add(word.lemma_)

    return list(unknown_words), num_words_sentence, num_unknown_words_sentence



def sentence_mine(chapter, known_words_series):

    #counts the number of real French words in the chapter
    num_words_chapter = 0
    #counts the number of real unknown French words in the chapter
    num_unknown_words_chapter = 0

    #break the individual chapter into sentences
    sentences = nltk.tokenize.sent_tokenize(chapter, language="french")

    #initialize a dataframe to contain the chapters' 1T sentences and
    #corresponding learned-words
    one_T_df = pd.DataFrame(columns=['sentence', 'word'])
    for sentence in sentences:

        #returns all unknown words from sentence
        rslt, num_words_sentence, num_unknown_words_sentence = get_T(sentence, known_words_series)

        num_words_chapter = num_words_chapter + num_words_sentence
        num_unknown_words_chapter = num_unknown_words_chapter + num_unknown_words_sentence

        #if T==1
        if len(rslt) == 1:

            unknown_word = rslt[0]

            row = pd.DataFrame({'sentence':[sentence], 'word':[unknown_word]})

            one_T_df = one_T_df.append(row)

    #many different sentences may be 1T and contain the new word
    #this keeps only the first 1T sentence where the word occurs
    one_T_df = one_T_df.drop_duplicates(subset=['word'])
    #reset index for good measure
    one_T_df = one_T_df.reset_index(drop=True)

    #extract learned words
    learned_words = list(one_T_df['word'].values)

    return learned_words, num_words_chapter, num_unknown_words_chapter

#updates the known words in the frequency lists
def update_known_words(learned_words, known_words_series, known_words):

    #update known_words
    for word in learned_words:

        #update the series-specific words to help find new unknown words in the series
        known_words_series.loc[known_words['word'] == word, 'known'] = 1
        #update the overall unknown words so that we can see how vocabulary is growing
        known_words.loc[known_words['word'] == word, 'known'] = 1


#returns the most common and least common words learned in the chapter
#also returns the median frequency score for the words learned
def learned_word_stats(learned_words, freq_list_series):

    learned_words_df = freq_list_series[freq_list_series['word'].isin(learned_words)]
    learned_words_df = learned_words_df.reset_index(drop=True)

    most_freq = learned_words_df.loc[0,'word']
    least_freq = learned_words_df.loc[len(learned_words_df)-1,'word']

    med_freq = np.median(learned_words_df['frequency'])

    return most_freq, least_freq, med_freq



def one_T_simulation(books_list, book_dfs, freq_list, freq_list_series, vocab_size_init=5000):



    known_words = freq_list.copy()

    #column for whether a word is known
    known_words['known'] = 0

    #say that the first 'vocab_size_init' words are known
    known_words.loc[:vocab_size_init-1, 'known'] = 1

    #limit to only words that occur in the books
    known_words_series = known_words.loc[freq_list_series.index]


    print('Initial Vocabulary Size:', vocab_size_init, 'words.')
    print('------------------------------------\n')




    #this dataframe will contain the results of our simulation
    learned_words_data = pd.DataFrame(columns=['Vocab_Init', 'Book', 'Chapter', 'Num_Words', 'Num_Unknown', 'Vocab_Before', 'Vocab_After', 'Most_Freq', 'Least_Freq', 'Med_Freq'])

    #iterate over each book
    for i in range(len(book_dfs)):

        print(books_list[i][:-3], i+1)

        book_df = book_dfs[i]

        #iterate through the chapters of the book
        for index in range(len(book_df)):

            #read a chapter
            chapter = book_df.loc[index, 'Text']

            #sentence mine the chapter to get learned words
            learned_words, num_words_chapter, num_unknown_words_chapter = sentence_mine(chapter, known_words_series)

            #num words known before sentence mining the chapter
            vocab_before = known_words['known'].sum()

            #update words known
            update_known_words(learned_words, known_words_series, known_words)


            #stats on the words learned
            most_freq, least_freq, med_freq = learned_word_stats(learned_words, freq_list_series)



            #num words known after sentence mining the chapter
            vocab_after = known_words['known'].sum()

            #update learned_words_data
            new_data = pd.DataFrame({'Vocab_Init':[vocab_size_init], 'Book':[i+1], 'Chapter':[index+1], 'Num_Words':[num_words_chapter], 'Num_Unknown':[num_unknown_words_chapter],
            'Vocab_Before':[vocab_before], 'Vocab_After':[vocab_after], 'Most_Freq':[most_freq], 'Least_Freq':[least_freq], 'Med_Freq':[med_freq]})
            learned_words_data = learned_words_data.append(new_data)


        print('-Vocabulary Size after reading Book:', vocab_after, '\n')

        #so that not all indices are 0
        learned_words_data = learned_words_data.reset_index(drop=True)

        #generate some extra statistics
        learned_words_data['Learned_Words'] = learned_words_data['Vocab_After'] - learned_words_data['Vocab_Before']
        learned_words_data['Perc_Unknown'] = 100 * learned_words_data['Num_Unknown'] / learned_words_data['Num_Words']
        learned_words_data['Learned_per_Unknown'] = 100 * learned_words_data['Learned_Words'] / learned_words_data['Num_Unknown']


    #return results
    return learned_words_data, known_words
