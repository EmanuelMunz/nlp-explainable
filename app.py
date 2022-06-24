import streamlit as st
from main import reviews_test,df_show, exp_DT, exp_SVM, exp_NB, exp_LR, exp_GB, classifier_DT,classifier_GB,classifier_LR,classifier_NB,classifier_SVM, y_test, test_vectors
from design import color_df
from ConfusionMatrix import createCM
from WordsImportance import make_importantWordsPlot
import streamlit.components.v1 as components


# import re
# import string
# import spacy
# import pickle
#########################################################
def main():
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.set_page_config(layout="wide")
    st.title('NLP - make it explainable')
    st.subheader('Sentiment Analysis')

    #explain field
    expander = st.expander("See explanation")
    expander.write("""
         This dashboard makes 5 different Machine Learning sentiment classifier explainable. 
         A Decision Tree Classifier(DT), Support Vector Machine(SVM), Naive Bayes(NB), Decision Tree(DT) and a Gradient Boosting (GB) classifier. 
         They are trained on amazon movie reviews. This dashboard uses LIME, a framework for XAI.""")

    #Seperator line
    st.markdown("""<hr style="height:10px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)



    #dict
    dictionary = {'negative': 0, 'neutral': 1, 'positive': 2}

    # Sidebar
    st.sidebar.title("Choose your visualization")

    # Data Frame
    st.subheader('Amazon Movie Reviews')

    #explain field
    expander = st.expander("See explanation")
    expander.write("""
         The table below shows a choosen number of amazon movie reviews from the test data set.
        It includes additional information whether the classifier predicted the underlying review correctly. By clicking on the
        headers of the columns you can filter the reviews.
     """)

    nb_rows = st.slider('Number of reviews to display', min_value=2, max_value=600, value=5, step=None,
                        format=None)
    #st.dataframe(df_show.head(nb_rows))
    st.dataframe(df_show.head(nb_rows).style.applymap(color_df, subset=['SVM_right','LR_right','GB_right','DT_right','NB_right']))

    #Seperator line
    st.markdown("""<hr style="height:10px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)

    if st.sidebar.checkbox("Single review analysis"):
        st.subheader('Single Review Analysis')

        # explain field
        expander = st.expander("See explanation")
        expander.write("""
             The Single Review Analysis can show how large of an impact single words have on each 
            review sentiment prediction. It also shows the Prediction probabilities for the three sentiment gradations 
            (positive, neutral, negative).""")

        revindex = st.sidebar.number_input('Insert index of checked review', min_value = 0, max_value = 599, value =  0, step =  1 )

        selsingleclf = st.sidebar.multiselect("Choose the classifiers", ["DT","SVM","NB","LR","GB"], key=1)
        #st.write("You selected", options)



        #def highlight pred_wrong:

        ##### Platzhalter Highlight Funktion

        #st.write('Text of the review: '+ df_show['body'][revindex])
        textbody = (df_show['body'][revindex])
        #st.subheader('Text of review '+str(revindex))
        #st.text_area(label='',value=textbody,disabled=True)

        # explain field
        expander = st.expander("Show review text")
        expander.write(textbody)


        reviews_subset = reviews_test[(["body"])]
        #reviews_subset = reviews_subset(revindex)
        #st.dataframe(reviews_subset.head())

        # Explainer single review
        if "DT" in selsingleclf:
            st.subheader('DT')
            components.html(exp_DT[revindex].as_html(),width = 1800, height = 250)

        if "SVM" in selsingleclf:
            st.subheader('SVM')
            components.html(exp_SVM[revindex].as_html(),width = 1800, height = 250)

        if "NB" in selsingleclf:
            st.subheader('NB')
            components.html(exp_NB[revindex].as_html(),width = 1800, height = 250)
        if "LR" in selsingleclf:
            st.subheader('LR')
            components.html(exp_LR[revindex].as_html(),width = 1800, height = 250)

        if "GB" in selsingleclf:
            st.subheader('GB')
            components.html(exp_GB[revindex].as_html(),width = 1800, height = 250)

        #Seperator line
        st.markdown("""<hr style="height:10px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)

    if st.sidebar.checkbox("Confusion Matrix"):
        st.subheader('Confusion Matrix')

        # explain field
        expander = st.expander("See explanation")
        expander.write("""
             The confusion matrix shows the accuracy of the classifier on the test set. It is useable to identify in what scenarios 
             the classifier perform well or bad.
         """)

        confmatr = st.sidebar.multiselect("Choose the classifiers", ["DT", "SVM", "NB", "LR", "GB"], key=2)
        col1, col2 = st.columns(2)
        # Wie viele clf sind ausgewählt in Dropdown?
        amount_selected_clf = len(confmatr)
        clf_list = confmatr
        dictionary2 = {'1': col1, '2': col2}
        array = []
        for i in range(0, amount_selected_clf):
            if i % 2 == 0:
                array.append('1')
            else:
                array.append('2')

        for i in range(0, amount_selected_clf):
            # Confusion Matrix
            if "DT" in confmatr:
                #dictionary2[array[0]].subheader('DecisionTree')
                dictionary2[array[0]].pyplot(createCM(classifier=classifier_DT,title = "Confusion Matrix DT" , y_test = y_test, test_vectors = test_vectors))
                # cut first index and drop clf
                array.pop(0)
                clf_list.remove('DT')
            elif "SVM" in confmatr:
                #dictionary2[array[0]].subheader('SupportVectorMachine')
                dictionary2[array[0]].pyplot(createCM(classifier=classifier_SVM, title="Confusion Matrix SVM", y_test=y_test,
                                     test_vectors=test_vectors))
                # cut first index and drop clf
                array.pop(0)
                clf_list.remove('SVM')
            elif "NB" in confmatr:
                #dictionary2[array[0]].subheader('NaiveBayes')
                dictionary2[array[0]].pyplot(createCM(classifier=classifier_NB, title="Confusion Matrix NB", y_test=y_test,
                                     test_vectors=test_vectors))
                # cut first index and drop clf
                array.pop(0)
                clf_list.remove('NB')
            elif "LR" in confmatr:
                #dictionary2[array[0]].subheader('LogisticRegression')
                dictionary2[array[0]].pyplot(createCM(classifier=classifier_LR, title="Confusion Matrix LR", y_test=y_test,
                                     test_vectors=test_vectors))
                # cut first index and drop clf
                array.pop(0)
                clf_list.remove('LR')

            elif "GB" in confmatr:
                #dictionary2[array[0]].subheader('GradientBoosting')
                dictionary2[array[0]].pyplot(createCM(classifier=classifier_GB, title="Confusion Matrix GB", y_test=y_test,
                                     test_vectors=test_vectors))
                # cut first index and drop clf
                array.pop(0)
                clf_list.remove('GB')
        #Seperator line
        st.markdown("""<hr style="height:10px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)

    if st.sidebar.checkbox("Multiple review analysis"):
        st.subheader('Multiple Review Analysis')

        # explain field
        expander = st.expander("See explanation")
        expander.write("""
             The multiple review analysis shows the most important words on each label and classifier. They are ordered from highest to lower  
             impact. The number in the bar shows how often the word influenced the sentiment prediction based on the test set.
         """)

        selmulclf = st.sidebar.multiselect("Choose the classifiers", ["DT", "SVM", "NB", "LR", "GB"], key=3)
        amountWords_show = st.slider('Number of words to display', min_value=1, max_value=20, value=5, step=None,
                            format=None)
        label = st.radio("Explaination of label", ('negative', 'neutral', 'positive'))
        st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)#show radio as row
        col1, col2 = st.columns(2)

        #Wie viele clf sind ausgewählt in Dropdown?
        amount_selected_clf = len(selmulclf)
        clf_list = selmulclf
        dictionary2 = {'1': col1, '2': col2}
        array = []
        for i in range(0, amount_selected_clf):
            if i % 2 == 0:
                array.append('1')
            else:
                array.append('2')

        for i in range(0,amount_selected_clf):
            # Explainer Multiple reviews
            if "DT" in selmulclf:
                dictionary2[array[0]].subheader('DT')
                dictionary2[array[0]].pyplot(make_importantWordsPlot('DT',amountWords_show,dictionary[label]))
                #cut first index and drop clf
                array.pop(0)
                clf_list.remove('DT')
            elif "SVM" in selmulclf:
                dictionary2[array[0]].subheader('SVM')
                dictionary2[array[0]].pyplot(make_importantWordsPlot('SVM',amountWords_show,dictionary[label]))
                # cut first index and drop clf
                array.pop(0)
                clf_list.remove('SVM')
            elif "NB" in selmulclf:
                dictionary2[array[0]].subheader('NB')
                dictionary2[array[0]].pyplot(make_importantWordsPlot('NB',amountWords_show,dictionary[label]))
                # cut first index and drop clf
                array.pop(0)
                clf_list.remove('NB')
            elif "LR" in selmulclf:
                dictionary2[array[0]].subheader('LR')
                dictionary2[array[0]].pyplot(make_importantWordsPlot('LR',amountWords_show,dictionary[label]))
                # cut first index and drop clf
                array.pop(0)
                clf_list.remove('LR')
            elif "GB" in selmulclf:
                dictionary2[array[0]].subheader('GB')
                dictionary2[array[0]].pyplot(make_importantWordsPlot('GB',amountWords_show,dictionary[label]))
                # cut first index and drop clf
                array.pop(0)
                clf_list.remove('GB')

        # Seperator line
        st.markdown("""<hr style="height:10px;border:none;color:#333;background-color:#333;" /> """,
                    unsafe_allow_html=True)
        #
        #     beeswarm_plot, ax = plt.subplots()
        #     ax = shap.plots.beeswarm(shap_values_multiple[:, :, 1], max_display=100)
        #
        #     st.pyplot(beeswarm_plot)
        #
        #     st.markdown("""The Beeswarmplot is a plot of all the SHAP values. The values are grouped by the features on the y-axis. For
        # each group, the colour of the points is determined by the value of the same feature (i.e. higher feature values
        # are redder). The features are ordered by the mean SHAP values.""")




if __name__ == '__main__':
    main()
