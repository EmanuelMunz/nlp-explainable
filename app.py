
import streamlit as st
from main import reviews_test, exp_DT, exp_SVM, exp_NB, exp_LR, exp_GB, classifier_DT,classifier_GB,classifier_LR,classifier_NB,classifier_SVM, y_test, test_vectors
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

    #dict
    dictionary = {'negative': 0, 'neutral': 1, 'positive': 2}

    # Sidebar
    st.sidebar.title("Choose your visualization")

    # ScatterPlot

    if st.sidebar.checkbox("Single review analysis"):

        revindex = st.sidebar.number_input('Insert index of checked review', min_value = 0, max_value = 599, value =  0, step =  1 )

        selsingleclf = st.sidebar.multiselect("Choose the classifiers", ["DT","SVM","NB","LR","GB"], key=1)
        #st.write("You selected", options)

        if st.checkbox('Show datatable'):
            nb_rows = st.slider('Number of rows to display', min_value=2, max_value=100, value=5, step=None,
                                format=None)
            st.dataframe(reviews_test.head(nb_rows))

        #def highlight pred_wrong:

        ##### Platzhalter Highlight Funktion

        st.write('The checked review has the index ', revindex)
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

    if st.sidebar.checkbox("Confusion Matrix"):
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
                dictionary2[array[0]].subheader('DT')
                dictionary2[array[0]].pyplot(createCM(classifier=classifier_DT,title = "Confusion Matrix DT" , y_test = y_test, test_vectors = test_vectors))
                # cut first index and drop clf
                array.pop(0)
                clf_list.remove('DT')
            elif "SVM" in confmatr:
                dictionary2[array[0]].subheader('SVM')
                dictionary2[array[0]].pyplot(createCM(classifier=classifier_SVM, title="Confusion Matrix DT", y_test=y_test,
                                     test_vectors=test_vectors))
                # cut first index and drop clf
                array.pop(0)
                clf_list.remove('SVM')
            elif "NB" in confmatr:
                dictionary2[array[0]].subheader('NB')
                dictionary2[array[0]].pyplot(createCM(classifier=classifier_NB, title="Confusion Matrix DT", y_test=y_test,
                                     test_vectors=test_vectors))
                # cut first index and drop clf
                array.pop(0)
                clf_list.remove('NB')
            elif "LR" in confmatr:
                dictionary2[array[0]].subheader('LR')
                dictionary2[array[0]].pyplot(createCM(classifier=classifier_LR, title="Confusion Matrix DT", y_test=y_test,
                                     test_vectors=test_vectors))
                # cut first index and drop clf
                array.pop(0)
                clf_list.remove('LR')

            elif "GB" in confmatr:
                dictionary2[array[0]].subheader('GB')
                dictionary2[array[0]].pyplot(createCM(classifier=classifier_GB, title="Confusion Matrix DT", y_test=y_test,
                                     test_vectors=test_vectors))
                # cut first index and drop clf
                array.pop(0)
                clf_list.remove('GB')

    if st.sidebar.checkbox("Multiple review analysis"):
        selmulclf = st.sidebar.multiselect("Choose the classifiers", ["DT", "SVM", "NB", "LR", "GB"], key=3)
        amountWords_show = st.slider('Number of words to display', min_value=1, max_value=10, value=5, step=None,
                            format=None)
        label = st.radio("Explaination for label", ('negative', 'neutral', 'positive'))
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

    ##########################################################