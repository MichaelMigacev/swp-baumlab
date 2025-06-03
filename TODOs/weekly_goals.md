## Weekly Goals and TODOs

##### This contains our weekly goals as well as the tracked worked on tasks

### Week 2

#### Goals:

DeepSynergy:

-   rewrite DeepSynergy to PyTorch
-   Understand input for DS (DeepSynergy) mostly (with biological details) and present to group

DeepSignalingFlow:

-   Understand input for DSF (DeepSignalingFlow) mostly (with biological details) and present to group
-   Understanding all the operations of DSF with corresponding code pieces and present to group
    -   copy DSF elements to our GitHub

Both:

-   understand validation standard (leave blablabla out) for data leakage free validation
    -   test with DS

#### Table Week 2

<table>
  <tr>
    <th>Names</th>
    <th>Tasks</th>
    <th>Time Taken</th>
  </tr>
  <tr>
    <td>Michael F.</td>
    <td>
        - Understood DeepSynergy<br>
        - Explained DeepSynergy in .ipynb `DeepSynergyExplanation.ipynb`<br>
        - Understood Validation, open issue of how the folds were created remains however. <br>
        - Found alternative pytorch implementation and researched feasability <br>
        - Investigated DS datasets <br>
        - Investigated Hyperparameter Search
    </td>
    <td> ~20h </td>
  </tr>
  <tr>
    <td>Sebastian</td>
    <td>
      - Reading into Neural Network basics <br>
      - Translated DeepSynergy from Tensorflow to Pytorch (REVIEW NEEDED) <br>
    </td>
    <td>20h</td>
  </tr>
  <tr>
    <td>Olha</td>
    <td>
        - Research into DeepSignalingFLow datasets <br>
        - Summary DeepSignalingFlow datasets
    </td>
    <td>20h</td>
  </tr>
  <tr>
    <td>Zhao</td>
    <td>
        - Researched various biological terms related to the topic Deepsynergy <br>
        - translated them into an easy-to-understand format <br>
        - Investigated DS
    </td>
    <td>20h</td>
  </tr>
  <tr>
    <td>Michael M.</td>
    <td>
        - SCRUM MASTER <br>
        - Made and organized repository <br>
        - Investigated DSF preprocessing <br>
    </td>
    <td>20h</td>
  </tr>
</table>

### Week 3

#### Goals:

DeepSynergy:



DeepSignalingFlow:



Both:



#### Table Week 3

<table>
  <tr>
    <th>Names</th>
    <th>Tasks</th>
    <th>Time Taken</th>
  </tr>
  <tr>
    <td>Michael F.</td>
    <td>
        Successfully implemented DeepSynergy and got expected MSE (though not visualized yet)
        Unsuccessfully tried to recreate hyperparameter search, but successfully recreated normalization and made it into a usable class function. (This took ages of trial and sadly mostly error and is still undone)
        Update here: GridSearch now works, but very slowly. Not sure its gonna run even overnight.
    </td>
    <td> ~25h </td>
  </tr>
  <tr>
    <td>Sebastian</td>
      <td>
        - Implementation Random Forest und Ridge Regression als Baselines
        - Aktuell auf Fehlersuche: Baselines performen auffällig besser als DeepSynergy
    </td>
    <td>22h</td>
  </tr>
  <tr>
    <td>Olha</td>
    <td>
      - Researched the preprocessing of data for DeepSignalingFlow (based on the original code of the authors) <br>
      - Recreated the data for training the model (it took about two hours to get just one out of four datasets processed) <br>
      - Started training the model on one ready dataset (but never finished due to computational resource limitations) -> will continue working on it in the next week
    </td>
    <td>20h</td>
  </tr>
  <tr>
    <td>Zhao</td>
    <td>
        Data-leakage checks in cross-validation -> there is no data leakage in leave drug combunation out method
        Discovered the provided labels.csv implements only leave-combination-out. 
        uploaded DEEPSYNERGY_THEORY file -> I refined my understanding and wrote up a clearer, more detailed explanation.
    </td>
    <td> 20h </td>
  </tr>
  <tr>
    <td>Michael M.</td>
    <td>
        - Researched DSF Preprocessing <br>
        - Recreated DSF Preprocessing in a detailed .ipynb file <br>
        - found a huge data leakage (depending on files used in final results by authors) 
    </td>
    <td>20h</td>
  </tr>
</table>

### Week 4


#### Table Week 4

<table>
  <tr>
    <th>Names</th>
    <th>Tasks</th>
    <th>Time Taken</th>
  </tr>
  <tr>
    <td>Michael F.</td>
    <td> Struggled a lot with the VPN on Linux and setting up the cluster. Manually had to try and try again and had a couple of inexplainable setbacks. Currently it runs, but does kick me out after a while.
    </td>
    <td> ~20h </td>
  </tr>
  <tr>
    <td>Sebastian</td>
      <td> Replicate and evaluate baseline models for drug synergy prediction on the cluster. Finally it worked and the results are as expected! Solid baseline for future work with deep learning models and XAI tools. I already got an Overview over SHAP and deepSHAP.
    </td>
    <td>20h</td>
  </tr>
  <tr>
    <td>Olha</td>
    <td>
    - Refined the code for training the DSF model ( + added functions to save the state of the training) <br>
    - Started trainig the DSF model on one of the datasets.
    </td>
    <td>10h</td>
  </tr>
  <tr>
    <td>Zhao</td>
    <td>
        - tried to finish data leakage check with my best efforts<br>
        - started with one of the explainability methods
    </td>
    <td> 20h </td>
  </tr>
  <tr>
    <td>Michael M.</td>
    <td>
        - continuation of full explanation for the GNN <br>
        - found some minor bugs
    </td>
    <td>~10h</td>
  </tr>
</table>

### Week 5

#### Goals:

DeepSynergy:



DeepSignalingFlow (notes): 

- Hyperparameter tuning (at least learning rate)
- 5-fold cross validation

#### Table Week 5

<table>
  <tr>
    <th>Names</th>
    <th>Tasks</th>
    <th>Time Taken</th>
  </tr>
  <tr>
    <td>Michael F.</td>
    <td> Started implementing SHAP. Currently SHAP Code seems to be working, but I tried working a lot with it just to notice after way too late that the shap code works but for some reason the model always predicts the same aka nothing. Currently trying to find out why. Once that works everything should be fine.
    </td>
    <td> ~22h </td>
  </tr>
  <tr>
    <td>Sebastian</td>
      <td>
    </td>
    <td>22h</td>
  </tr>
  <tr>
    <td>Olha</td>
    <td>
    - Revised the code for training </br>
     - Trained models (solved the issues in process)
    </td>
    <td>20h</td>
  </tr>
  <tr>
    <td>Zhao</td>
    <td>
        - tried implement integrated gradients as explainability method for deepsynergy </br>
    </td>
    <td> 20h </td>
  </tr>
  <tr>
    <td>Michael M.</td>
    <td>
        - preparation for training <br>
        - complete explanation of the GNN (line for line)
    </td>
    <td>~25h</td>
  </tr>
</table>

### Week 6

#### Goals:

DeepSynergy:



DeepSignalingFlow (notes): 



#### Table Week 6

<table>
  <tr>
    <th>Names</th>
    <th>Tasks</th>
    <th>Time Taken</th>
  </tr>
  <tr>
    <td>Michael F.</td>
    <td> 
    </td>
    <td> ~20h </td>
  </tr>
  <tr>
    <td>Sebastian</td>
      <td>
    </td>
    <td>22h</td>
  </tr>
  <tr>
    <td>Olha</td>
    <td>
    </td>
    <td>20h</td>
  </tr>
  <tr>
    <td>Zhao</td>
    <td>
        - 
    </td>
    <td> 20h </td>
  </tr>
  <tr>
    <td>Michael M.</td>
    <td>
        - training 5 fold-cross-validation <br>
        - look into explainability
    </td>
    <td>~25h</td>
  </tr>
</table>
