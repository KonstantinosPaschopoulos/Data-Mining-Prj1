from wordcloud import WordCloud, STOPWORDS
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

def main():
    stop_words = set(STOPWORDS)
    stop_words.update(ENGLISH_STOP_WORDS)
    #extra stop words
    extra_words=["said","say","seen","come","end","came","year","years","new","saying"]
    stop_words = ENGLISH_STOP_WORDS.union(extra_words)

    df = pd.read_csv('train_set.csv', sep='\t')

    cat_politics = []
    cat_film = []
    cat_football = []
    cat_business = []
    cat_technology = []
    #store the content for each category
    for index in range(len(df.Category)):
        cat = df.Category[index]
        if cat == "Politics":
            cat_politics.append(df.Content[index])
        elif cat == "Film":
            cat_film.append(df.Content[index])
        elif cat == "Football":
            cat_football.append(df.Content[index])
        elif cat == "Business":
            cat_business.append(df.Content[index])
        elif cat == "Technology":
            cat_technology.append(df.Content[index])

    str_pol = ''.join(cat_politics)
    str_fil = ''.join(cat_film)
    str_foo = ''.join(cat_football)
    str_bus = ''.join(cat_business)
    str_tec = ''.join(cat_technology)

    #produce wordcloud for each category
    cloud = WordCloud(background_color="white", mode = "RGB", stopwords = stop_words, width=1920, height=1080)
    w = cloud.generate(str_pol)
    plt.figure()
    plt.title("Politics")
    plt.imshow(w)
    plt.axis("off")
    plt.savefig('Politics.png')


    w = cloud.generate(str_fil)
    plt.figure()
    plt.title("Film")
    plt.imshow(w)
    plt.axis("off")
    plt.savefig('Film.png')

    w = cloud.generate(str_foo)
    plt.figure()
    plt.imshow(w)
    plt.title("Football")
    plt.axis("off")
    plt.savefig('Football.png')

    w = cloud.generate(str_bus)
    plt.figure()
    plt.imshow(w)
    plt.title("Business")
    plt.axis("off")
    plt.savefig('Business.png')

    w = cloud.generate(str_tec)
    plt.figure()
    plt.imshow(w)
    plt.title("Technology")
    plt.axis("off")
    plt.savefig('Technology.png')




if __name__ == "__main__":
    main()
