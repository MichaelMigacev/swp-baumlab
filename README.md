# Repository of Softwareproject at Baumlab

### Replicating and Analyzing existing Machine Learning Methods for Drug Synergies with respects to explainability

#### Chosen Projects:

##### DeepSynergy [Link](https://doi.org/10.1093/bioinformatics/btx806)

##### DeepSignalingFlow [Link](https://doi.org/10.1038/s41540-024-00421-w)

## Working with our Github Repository


### Setup

1. Install git (mandatory)
2. Install VS Code (for ease of use)
3. Clone the repository use `git clone`

### Usage

1. Sync with main
```
git checkout main
git pull origin main
```
2. Create a new branch
```
git checkout -b name-of-your-branch
```
3. Write your code on this branch
   
   Keep Code small for reviewers(100 - 200 lines)

4. stage changes & commit
```
git add .
git commit -m "Description of changes" 
```
5. Push to GitHub
```
git push origin your-branch-name
```
6. Open a Pull Request
  1. Go to GitHub
  2. Click "Compare & Pull Request"
  3. Summarize Your Changes
  4. Reqeust at least one reviewer

7. Wait for Approval
   
8. Merge (Squash & Merge) 

9. Delete old branch
   with a button in GitHub
   like this locally:
   ```
   git checkout main
   git branch -d name-of-your-branch
   ```
10. Sync Regularly
   ```
   git checkout main
   git pull origin main
   git checkout your-branch
   git merge main
   ```
