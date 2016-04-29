from django.shortcuts import render, render_to_response, get_object_or_404
from django.template import RequestContext
from django.http import HttpResponseRedirect
from django.core.urlresolvers import reverse

from myproject.myapp.models import Document
from myproject.myapp.forms import DocumentForm
import pandas
import copy

from django.contrib.auth.decorators import login_required

from myproject import settings



@login_required
def home(request):
    import os
    # Handle file upload
    if request.method == 'POST':
        form = DocumentForm(request.POST, request.FILES)
        if form.is_valid():
            newdoc = Document(docfile=request.FILES['docfile'])
            newdoc.save()
            # Redirect to the document list after POST
            return HttpResponseRedirect(reverse('myproject.myapp.views.home'))
    else:
        form = DocumentForm()  # A empty, unbound form

    # Load documents for the list page
    documents = Document.objects.all()
    for document in documents:
        document.docfile.documentName = os.path.basename(document.docfile.name)


    # Render list page with the documents and the form
    return render_to_response(
        'home.html',
        {'documents': documents, 'form': form},
        context_instance=RequestContext(request)
    )


class Classifer(object):

    def __init__(self, filename):
        import pandas
        import os
        from sklearn import preprocessing
        le = preprocessing.LabelEncoder()
        self.path = filename
        self.documentName = os.path.basename(self.path)
        self.dataset = pandas.read_csv(filename)
        self.rows = len(self.dataset) 
        self.cols = len(self.dataset.columns.values)
        self.colNames = list(self.dataset.columns.values)
        



    def setClass(self, target):
        from sklearn import preprocessing
        import pandas as pd
        import copy
        le = preprocessing.LabelEncoder()
        self.codebook = dict()
        self.target = self.dataset[target]
        self.dataCol = copy.copy(self.colNames)
        self.dataCol.remove(target)
        self.data = self.dataset[self.dataCol]
        self.classLevel = []
        # Because sklearn can only deal with numeric values, we have to convert
        # strings into numbers.
        if (self.target.dtype != "int64" and self.target.dtype != "float64"):
            le.fit(self.target)
            code = le.transform(le.classes_)
            for i in code:
                self.codebook[i] = le.classes_[i]
                self.classLevel.append(le.classes_[i])
            for r in range(self.rows):
                ind = self.target[r]
                self.target[r] = list(le.classes_).index(ind)
            self.target = self.target.astype(int)
        else:
            classLevel = self.target.unique()
            for level in classLevel:
                self.classLevel.append(str(level))
        tempData = pd.DataFrame()
        tempCol = None
        for col in self.dataCol:
            if (self.data[col].dtype != "int64" and 
                self.data[col].dtype != "float64"):
                tempCol = pd.get_dummies(self.data[col]).astype("int64")
                tempData = pd.concat([tempData, tempCol], axis = 2)
            else:
                tempData = pd.concat([tempData, self.data[col]], axis = 2)
        self.data = tempData



    def getDTMetrics(self, criterion='gini', splitter='best', max_depth=None, 
                     min_samples_split=2, min_samples_leaf=1, 
                     min_weight_fraction_leaf=0.0, max_features=None, 
                     random_state=None, max_leaf_nodes=None, class_weight=None,
                     presort=False):
        from sklearn import cross_validation
        from sklearn.metrics import classification_report
        from sklearn.tree import DecisionTreeClassifier
        DT = DecisionTreeClassifier(criterion, splitter, max_depth, 
                                    min_samples_split, min_samples_leaf, 
                                    min_weight_fraction_leaf, max_features, 
                                    random_state, max_leaf_nodes, class_weight,
                                    presort)
        self.DTModel = DT.fit(self.data, self.target)
        self.DTPrediction = self.DTModel.predict(self.data)
        self.DTClReport = classification_report(self.target, self.DTPrediction,
                                                target_names = self.classLevel)

        self.TenDTCVAccu = cross_validation.cross_val_score(DT, self.data, 
                                                         self.target, 
                                                         scoring='accuracy', 
                                                         cv = 10)
        self.DTCVAccu = sum(self.TenDTCVAccu)/10

    def getDTGrapchis(self):
        from sklearn.tree import export_graphviz
        from subprocess import call
        import os
        outPutPath = settings.MEDIA_ROOT
        mediaFiles = os.listdir(path = outPutPath)
        for item in mediaFiles:
            tempPath = os.path.join(outPutPath,item)
            if (os.path.isdir(tempPath) == False):
                if (item.startswith("tree")):
                    os.remove(tempPath)
        treePath = os.path.join(outPutPath,'tree.dot')
        export_graphviz(self.DTModel,out_file=treePath, 
                                     feature_names = list(self.data.columns.values),
                                     class_names = self.classLevel,
                                     filled = True, rounded = True,
                                     special_characters = True)
        pngPath = os.path.join(outPutPath,'tree.png')
        call(['dot','-Tpng',treePath,'-o',pngPath])

    def getSVMMetrics(self):
        from sklearn import cross_validation
        from sklearn.metrics import classification_report
        from sklearn import svm
        SVM = svm.SVC(kernel = "linear")
        self.SVMModel = SVM.fit(self.data, self.target)
        self.SVMPrediction = self.SVMModel.predict(self.data)
        self.SVMClReport = classification_report(self.target,self.SVMPrediction,
                                                 target_names = self.classLevel)

        self.TenSVMCVAccu = cross_validation.cross_val_score(SVM, self.data, 
                                                             self.target, 
                                                             scoring='accuracy', 
                                                             cv = 10)
        self.SVMCVAccu = sum(self.TenSVMCVAccu)/10

    def getNBMetrics(self):
        from sklearn import cross_validation
        from sklearn.metrics import classification_report
        from sklearn.naive_bayes import GaussianNB
        NB = GaussianNB()
        self.NBModel = NB.fit(self.data, self.target)
        self.NBPrediction = self.NBModel.predict(self.data)
        self.NBClReport = classification_report(self.target,self.NBPrediction,
                                                target_names = self.classLevel)

        self.TenNBCVAccu = cross_validation.cross_val_score(NB, self.data, 
                                                            self.target, 
                                                            scoring='accuracy', 
                                                            cv = 10)
        self.NBCVAccu = sum(self.TenNBCVAccu)/10


    def getRFMetrics(self, n_estimators=10, criterion='gini', max_depth=None, 
                     min_samples_split=2, min_samples_leaf=1, 
                     min_weight_fraction_leaf=0.0, max_features='auto', 
                     max_leaf_nodes=None, bootstrap=True, oob_score=False, 
                     n_jobs=1, random_state=None, verbose=0, warm_start=False, 
                     class_weight=None):
        from sklearn import cross_validation
        from sklearn.metrics import classification_report
        from sklearn.ensemble import RandomForestClassifier
        RF = RandomForestClassifier(n_estimators, criterion, max_depth, 
                                    min_samples_split, min_samples_leaf, 
                                    min_weight_fraction_leaf, max_features, 
                                    max_leaf_nodes, bootstrap, oob_score, 
                                    n_jobs, random_state, verbose, warm_start, 
                                    class_weight)
        self.RFModel = RF.fit(self.data, self.target)
        self.RFPrediction = self.RFModel.predict(self.data)
        self.RFClReport = classification_report(self.target,self.RFPrediction,
                                                target_names = self.classLevel)

        self.TenRFCVAccu = cross_validation.cross_val_score(RF, self.data, 
                                                            self.target, 
                                                            scoring='accuracy', 
                                                            cv = 10)
        self.RFCVAccu = sum(self.TenRFCVAccu)/10

    def getROC(self, bestModel):
        import numpy as np
        import matplotlib
        import matplotlib.pyplot as plt
        from sklearn.metrics import roc_curve, auc
        from sklearn.preprocessing import label_binarize
        from sklearn.multiclass import OneVsRestClassifier
        from scipy import interp
        from sklearn import svm
        from sklearn import preprocessing
        le = preprocessing.LabelEncoder()
        import os

        # First, binarize the output
        self.bTarget = label_binarize(self.target, 
                                      classes=list(self.target.unique()))
        if (len(self.classLevel) == 2):
            if (list(self.target.unique()) == [0, 1]):
                tempBTarget = []
                for i in self.bTarget:
                    i = [1 - i[0]] + list(i)
                    tempBTarget.append(i)
                self.bTarget = np.asarray(tempBTarget)
            elif (list(self.target.unique()) == [1, 0]):
                tempBTarget = []
                for i in self.bTarget:
                    i = list(i) + [1 - i[0]] 
                    tempBTarget.append(i)
                self.bTarget = np.asarray(tempBTarget)
            else:
                return


        # Learn to predict each class against the other
        if (bestModel == "Naive Bayes"):
            self.score = self.NBModel.predict_proba(self.data)
        elif (bestModel == "Support Vector Machine"):
            newSVM = svm.SVC(kernel='linear',probability = True)
            newSVMModel = newSVM.fit(self.data, self.target)
            self.score = newSVMModel.predict_proba(self.data)
        elif (bestModel == "Random Forest"):
            self.score = self.RFModel.predict_proba(self.data)
        elif (bestModel == "Decision Tree"):
            self.score = self.DTModel.predict_proba(self.data)
        else:
            return

        # Compute ROC curve and ROC area for each class
        self.fpr = dict()
        self.tpr = dict()
        self.rocAuc = dict()

        for i in range(len(self.classLevel)):
            self.fpr[i], self.tpr[i], _ = roc_curve(self.bTarget[:,i], 
                                                    self.score[:,i])
            self.rocAuc[i] = auc(self.fpr[i], self.tpr[i])

        self.fpr["micro"], self.tpr["micro"], _ = roc_curve(self.bTarget.ravel(), 
                                                            self.score.ravel())
        self.rocAuc["micro"] = auc(self.fpr["micro"], self.tpr["micro"])


        # Compute macro-average ROC curve and ROC area
        # First aggregate all false positive rates
        self.allFpr = np.unique(np.concatenate([self.fpr[i] for i in 
                                                range(len(self.classLevel))]))

        # Then interpolate all ROC curves at this points
        self.meanTpr = np.zeros_like(self.allFpr)
        for i in range(len(self.classLevel)):
            self.meanTpr += np.interp(self.allFpr, self.fpr[i], self.tpr[i])

        # Finally average it and compute AUC
        self.meanTpr /= len(self.classLevel)

        self.fpr["macro"] = self.allFpr
        self.tpr["macro"] = self.meanTpr
        self.rocAuc["macro"] = auc(self.fpr["macro"], self.tpr["macro"])
   
        
        # Delete the old png
        outPutPath = settings.MEDIA_ROOT
        mediaFiles = os.listdir(path = outPutPath)
        for item in mediaFiles:
            tempPath = os.path.join(outPutPath,item)
            if (os.path.isdir(tempPath) == False):
                if (item.startswith("roc")):
                    os.remove(tempPath)
        rocPath = os.path.join(outPutPath,'roc.png')

        # Plot all ROC curves
        matplotlib.use('Agg')
        plt.plot(self.fpr["micro"], self.tpr["micro"],
                 label='micro-average ROC curve (area = %0.2f)'
                       %(self.rocAuc["micro"]),
                 linewidth=2)

        plt.plot(self.fpr["macro"], self.tpr["macro"],
                 label='macro-average ROC curve (area = %0.2f)'
                       %(self.rocAuc["macro"]),
                 linewidth=2)

        for i in range(len(self.classLevel)):
            plt.plot(self.fpr[i], self.tpr[i], 
                     label='ROC curve of class %s (area = %0.2f)'
                     % (self.classLevel[i], self.rocAuc[i]))

        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic of our best model: %s'
                  % (bestModel))
        plt.legend(loc="lower right")
        plt.savefig(rocPath)
        plt.close('all')




def formatReport(input):
    result = []
    for line in input:
        if (line != ""):
            tempLine = line.split()
            if (len(tempLine) == 4):
                tempLine = [" "] + tempLine
                result.append(tempLine)
            elif (len(tempLine) == 7):
                tempLine = ["avg/total"] + tempLine[-4:]
                result.append(tempLine)
            else:
                result.append(tempLine)
    return result

def classification(file, target):
    file.setClass(target)
    file.getDTMetrics() 
    file.getDTGrapchis()
    file.getSVMMetrics()
    file.getNBMetrics()
    file.getRFMetrics()
    tempDTResult = str(file.DTClReport).splitlines()
    tempSVMResult = str(file.SVMClReport).splitlines()
    tempNBResult = str(file.NBClReport).splitlines()
    tempRFResult = str(file.RFClReport).splitlines()
    NBResult = formatReport(tempNBResult)
    SVMResult = formatReport(tempSVMResult)
    RFResult = formatReport(tempRFResult)
    DTResult = formatReport(tempDTResult)
    BestModel = None 
    BestScore = 0
    allResults = [file.NBCVAccu, file.SVMCVAccu, 
                  file.RFCVAccu, file.DTCVAccu]
    ModelNames = ["Naive Bayes", "Support Vector Machine", 
                  "Random Forest", "Decision Tree"]
    for i in range(4):
        if (allResults[i] > BestScore):
            BestModel = ModelNames[i]
            BestScore = allResults[i]
    if (allResults[0] == allResults[1] == allResults[2] == allResults[3]):
        BestModel = "Fuck! They are all the same!"
    file.getROC(BestModel)
    return (DTResult, SVMResult, NBResult, RFResult, BestScore, BestModel)
    

def explor(file, target):
    import matplotlib
    import matplotlib.pyplot as plt
    import os
    import seaborn as sns
    # Delete the old png
    outPutPath = settings.MEDIA_ROOT
    mediaFiles = os.listdir(path = outPutPath)
    for item in mediaFiles:
        tempPath = os.path.join(outPutPath,item)
        if (os.path.isdir(tempPath) == False):
            if (item.startswith("expl") or item.startswith("pairplot")):
                os.remove(tempPath)
    explPathOne = os.path.join(outPutPath,'expl.png')
    explPathTwo = os.path.join(outPutPath,'pairplot.png')
    matplotlib.use('Agg')
    sns.set(style="whitegrid", color_codes=True)
    if (file.dataset[target].dtype != "int64" 
        and file.dataset[target].dtype != "float64"):
        g = sns.countplot(x=target, data=file.dataset, palette="Greens_d")
        g.figure.subplots_adjust(bottom=0.4)
        for item in g.xaxis.get_major_ticks():
            item.label.set_fontsize(8)
            item.label.set_rotation(90)
        plt.tight_layout()
        
    else:
        file.dataset.hist(column = target)
    plt.savefig(explPathOne)
    plt.close('all')
    datatypes = set()
    for col in file.colNames:
        datatypes.add(str(file.dataset[col].dtype))
    if ((len(file.colNames) > 10) or (('int64' not in datatypes) and 
        ('float64' not in datatypes))):
        matplotlib.use('Agg')
        fig = plt.figure()
        ax = fig.add_subplot(111)
        fig = fig.subplots_adjust(top=0.85)
        ax.text(0.2, 0.8, 'Sometimes, you just cannot get what you want.', 
                style='italic',bbox={'facecolor':'red', 'alpha':0.5, 'pad':10})
        ax.text(0.2, 0.6, 'There are two possible reasons:', 
                style='italic',bbox={'facecolor':'blue', 'alpha':0.5, 'pad':5})
        ax.text(0.2, 0.5, '1. You have too many columns; ', 
                style='italic',bbox={'facecolor':'blue', 'alpha':0.5, 'pad':5})
        ax.text(0.2, 0.4, '2. All your columns are categorial variables', 
                style='italic',bbox={'facecolor':'blue', 'alpha':0.5, 'pad':5})
        ax.text(0.2, 0.3, 'Unhappy? Go to use other website, please!', 
                style='italic',bbox={'facecolor':'yellow','alpha':0.5, 'pad':5})
    else:
        matplotlib.use('Agg')
        sns.set()
        g = sns.pairplot(file.dataset, hue = target)
        g.fig.subplots_adjust(top=0.8, right = 0.8)
        g.fig.suptitle('Pair Plots of All Numberic Variables', 
                        fontsize=20,color="r",alpha=0.5)
        # plt.legend(prop={'size':6})  
    plt.savefig(explPathTwo)
    plt.close('all')



def getDefaultResult(request, file, DIYMethod):
    if (DIYMethod == "Decision Tree"):
        file.getDTMetrics()
        file.getDTGrapchis()
        tempDTDIYResult = str(file.DTClReport).splitlines()
        defaultDTResult = formatReport(tempDTDIYResult)
        defaultDTCVAccu = file.DTCVAccu
        return (defaultDTResult, defaultDTCVAccu)
    elif (DIYMethod == "Random Forest"):
        file.getRFMetrics()
        tempRFDIYResult = str(file.RFClReport).splitlines()
        defaultRFResult = formatReport(tempRFDIYResult)
        defaultRFCVAccu = file.RFCVAccu
        return (defaultRFResult, defaultRFCVAccu)
    elif (DIYMethod == "Support Vector Machine"):
        file.getSVMMetrics()
        tempSVMDIYResult = str(file.SVMClReport).splitlines()
        defaultSVMResult = formatReport(tempSVMDIYResult)
        defaultSVMCVAccu = file.SVMCVAccu
        return (defaultSVMResult, defaultSVMCVAccu)


def dosomething(request):
    if (request.method == 'POST'):
        document_id = request.POST.get('document_id')
        dataset = get_object_or_404(Document, id=document_id)
        thePath = dataset.docfile.path
        file = Classifer(thePath)
        if (request.POST.get('action') == "Classification"):
            target = request.POST.get('dropdownClass')
            (DTResult, SVMResult, NBResult, 
             RFResult, BestScore, BestModel) = classification(file, target)
            return render(request, 
                          'classification.html',
                          {'DTResult':DTResult, 'SVMResult':SVMResult, 
                          'NBResult':NBResult, 'RFResult':RFResult,
                          'BestScore':BestScore, 'BestModel':BestModel,
                          'target':target, 'file':file}
                         )
        elif (request.POST.get('action') == "Exploration"):
            target = request.POST.get('dropdownClass')
            explor(file, target)
            return render(request, 'exploration.html', 
                          {'file':file, 'document_id':document_id, 
                           'target':target})
        elif (request.POST.get('action') == "DIY"):
            target = request.POST.get('dropdownClass')
            file.setClass(target)
            DIYMethod = request.POST.get('DIYMethod')
            (Result, CVAccu) = getDefaultResult(request, file, DIYMethod)
            return render(request, 'DIY.html',
                          {'file':file, 'document_id':document_id, 
                           'DIYMethod':DIYMethod,
                           'Result':Result, 
                           'CVAccu':CVAccu,
                           'target':target}
                         )

        else:
            return render(request, 'load.html')


def exploration(request):
    import matplotlib
    import matplotlib.pyplot as plt
    import os
    if (request.method == 'POST'):
        document_id = request.POST.get('document_id')
        dataset = get_object_or_404(Document, id=document_id)
        thePath = dataset.docfile.path
        file = Classifer(thePath)
        target = request.POST.get('dropdownClass')
        explor(file, target)
        return render(request, 'exploration.html', 
                  {'file':file, 'document_id':document_id})

def convert(stuff):
    result = None
    if (stuff == "None"):
        return result
    else:
        if stuff.isdigit():
            result = int(stuff)
        elif stuff == "True":
            result = True
        elif stuff == "False":
            result = False
        else:
            try:
                result = float(stuff)
            except:
                result = stuff
        return result

def DIY(request):
    if (request.method == 'POST'):
        document_id = request.POST.get('document_id')
        dataset = get_object_or_404(Document, id=document_id)
        thePath = dataset.docfile.path
        file = Classifer(thePath)
        target = request.POST.get('dropdownClass')
        file.setClass(target)
        if (request.POST.get('action') == "Classification"):
            DIYMethod = request.POST.get('DIYMethod')
            if (DIYMethod == "Decision Tree"):
                criterion = request.POST.get('criterion')
                splitter = request.POST.get('splitter')
                max_features = convert(request.POST.get('max_features'))
                max_depth = convert(request.POST.get('max_depth'))
                min_samples_split = convert(request.POST.get('min_samples_split'))
                min_samples_leaf = convert(request.POST.get('min_samples_leaf'))
                min_weight_fraction_leaf = convert(request.POST.get('min_weight_fraction_leaf'))
                max_leaf_nodes = convert(request.POST.get('max_leaf_nodes'))
                class_weight = convert(request.POST.get('class_weight'))
                random_state = convert(request.POST.get('random_state'))
                presort = convert(request.POST.get('presort'))
                file.getDTMetrics(criterion, splitter, max_depth, 
                                  min_samples_split, min_samples_leaf, 
                                  min_weight_fraction_leaf, max_features, 
                                  random_state, max_leaf_nodes, class_weight,
                                  presort)
                file.getDTGrapchis()
                tempDIYDTResult = str(file.DTClReport).splitlines()
                DIYDTResult = formatReport(tempDIYDTResult)
                DIYDTCVAccu = file.DTCVAccu
                return render(request, 'DIY.html',
                              {'file':file, 'document_id':document_id, 
                               'DIYMethod':DIYMethod,
                               'Result':DIYDTResult, 
                               'CVAccu':DIYDTCVAccu,
                               'target':target}
                             )
            elif (DIYMethod == "Random Forest"):
                n_estimators = convert(request.POST.get('n_estimators'))
                criterion = convert(request.POST.get('criterion'))
                max_features = convert(request.POST.get('max_features'))
                max_depth = convert(request.POST.get('max_depth'))
                min_samples_split = convert(request.POST.get('min_samples_split'))
                min_samples_leaf = convert(request.POST.get('min_samples_leaf'))
                min_weight_fraction_leaf = convert(request.POST.get('min_weight_fraction_leaf'))
                max_leaf_nodes = convert(request.POST.get('max_leaf_nodes'))
                class_weight = convert(request.POST.get('class_weight'))
                random_state = convert(request.POST.get('random_state'))
                bootstrap = convert(request.POST.get('bootstrap'))
                oob_score = convert(request.POST.get('oob_score'))
                n_jobs = convert(request.POST.get('n_jobs'))
                verbose = convert(request.POST.get('verbose'))
                warm_start = convert(request.POST.get('warm_start'))
                print(n_estimators)
                file.getRFMetrics(n_estimators=n_estimators, criterion=criterion, 
                                   max_depth=max_depth, min_samples_split=min_samples_split, 
                                   min_samples_leaf=1, 
                                   min_weight_fraction_leaf=min_weight_fraction_leaf, 
                                   max_features='auto', max_leaf_nodes=None, 
                                   bootstrap=bootstrap, oob_score=oob_score, 
                                   n_jobs=n_jobs, random_state=None, verbose=0, 
                                   warm_start=False, class_weight=None)
                tempDIYRFResult = str(file.RFClReport).splitlines()
                DIYRFResult = formatReport(tempDIYRFResult)
                DIYRFCVAccu = file.RFCVAccu
                return render(request, 'DIY.html',
                              {'file':file, 'document_id':document_id, 
                               'DIYMethod':DIYMethod,
                               'Result':DIYRFResult, 
                               'CVAccu':DIYRFCVAccu}
                             )
            elif (DIYMethod == "Support Vector Machine"):
                C = convert(request.POST.get('C'))
                kernel = convert(request.POST.get('kernel'))
                degree = convert(request.POST.get('degree'))
                gamma = convert(request.POST.get('gamma'))
                coef0 = convert(request.POST.get('coef0'))
                probability = convert(request.POST.get('probability'))
                shrinking = convert(request.POST.get('shrinking'))
                tol = convert(request.POST.get('tol'))
                cache_size = convert(request.POST.get('cache_size'))
                class_weight = convert(request.POST.get('class_weight'))
                verbose = convert(request.POST.get('verbose'))
                max_iter = convert(request.POST.get('max_iter'))
                decision_function_shape = convert(request.POST.get('decision_function_shape'))
                random_state = convert(request.POST.get('random_state'))
                from sklearn import cross_validation
                from sklearn.metrics import classification_report
                from sklearn import svm
                if (kernel == "rbf"):
                    SVM = svm.SVC(C=C, kernel=kernel, gamma=gamma, 
                                  shrinking=shrinking, 
                                  probability=probability, tol=tol, 
                                  cache_size=cache_size, 
                                  class_weight=class_weight, 
                                  verbose=verbose, max_iter=max_iter, 
                                  decision_function_shape=decision_function_shape, 
                                  random_state=random_state)
                elif (kernel == "sigmoid"):
                    SVM = svm.SVC(C=C, kernel=kernel, gamma=gamma, coef0=coef0,
                                  shrinking=shrinking, probability=probability, 
                                  tol=tol, cache_size=cache_size, 
                                  class_weight=class_weight, 
                                  verbose=verbose, max_iter=max_iter, 
                                  decision_function_shape=decision_function_shape, 
                                  random_state=random_state)
                elif (kernel == "poly"):
                    SVM = svm.SVC(C=C, kernel=kernel, degree=degree, gamma=gamma, 
                                  coef0=coef0, shrinking=shrinking, 
                                  probability=probability, tol=tol, 
                                  cache_size=cache_size, 
                                  class_weight=class_weight, 
                                  verbose=verbose, max_iter=max_iter, 
                                  decision_function_shape=decision_function_shape, 
                                  random_state=random_state)
                else:
                    SVM = svm.SVC(C=C, kernel=kernel, shrinking=shrinking, 
                                  probability=probability, 
                                  tol=tol, cache_size=cache_size, 
                                  class_weight=class_weight, verbose=verbose, 
                                  max_iter=max_iter, 
                                  decision_function_shape=decision_function_shape, 
                                  random_state=random_state)


                SVMModel = SVM.fit(file.data, file.target)
                SVMPrediction = SVMModel.predict(file.data)
                SVMClReport = classification_report(file.target, SVMPrediction,
                                                    target_names = file.classLevel)

                TenSVMCVAccu = cross_validation.cross_val_score(SVM, file.data, 
                                                                file.target, 
                                                                scoring='accuracy', 
                                                                cv = 10)
                SVMCVAccu = sum(TenSVMCVAccu)/10
                tempDIYSVMResult = str(SVMClReport).splitlines()
                DIYSVMResult = formatReport(tempDIYSVMResult)
                return render(request, 'DIY.html',
                              {'file':file, 'document_id':document_id, 
                               'DIYMethod':DIYMethod,
                               'Result':DIYSVMResult, 
                               'CVAccu':SVMCVAccu,
                               'target':target}
                             )

        elif (request.POST.get('action') == "Channge Method"):
            DIYMethod = request.POST.get('ChangeDIYMethod')
            (Result, CVAccu) = getDefaultResult(request, file, DIYMethod)
            return render(request, 'DIY.html',
                          {'file':file, 'document_id':document_id, 
                           'DIYMethod':DIYMethod,
                           'Result':Result, 
                           'CVAccu':CVAccu,
                           'target':target}
                         )



def load(request):
    if request.method == 'POST':
        document_id = request.POST['document']
        dataset = get_object_or_404(Document, id=document_id)
        thePath = dataset.docfile.path
        file = Classifer(thePath)
        return render(request, 
                'load.html',
                {'file': file, 'document_id':document_id}
            )

def tutorial(request):
    if request.method == 'GET':
        return render(request, 
                'tutorial.html',
            )

def about(request):
    if request.method == 'GET':
        return render(request, 
                'about.html',
            )


def contact(request):
    if request.method == 'GET':
        return render(request, 
                'contact.html',
            )

