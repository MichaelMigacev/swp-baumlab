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
        Implementation Random Forest und Ridge Regression als Baselines
        Verwendete Nested Cross-Validation mit GridSearch
        Arbeitete mit tanh-normalisierten DeepSynergy-Daten
        Aktuell auf Fehlersuche: Baselines performen auffällig besser als DeepSynergy
    <td>
    </td>
    <td>22h</td>
  </tr>
  <tr>
    <td>Olha</td>
    <td>
    </td>
    <td></td>
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
    </td>
    <td></td>
  </tr>
</table>
