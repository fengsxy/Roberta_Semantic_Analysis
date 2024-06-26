This codebase is tailored for training a RoBERTa model on Amazon product review texts, with the data being segmented into two categories to serve distinct purposes: binary and five-way classifications. To streamline the debugging process, we've included a smaller dataset. Detailed analyses and outcomes are documented in the [Data_Processing_Analysis.ipynb](https://github.com/fengsxy/Robertaforsenmantic/blob/main/Data_Processing_Analysis.ipynb) file. Training logs are systematically archived in the `./logs` directory. For access to the trained model, please refer to the following links:

To download the trained RoBERTa model, please click on the following Google Drive link:
[Download Trained Model (Google Drive)](https://drive.google.com/drive/folders/1fXSWaMkOE5SRYzMmqKAcY80lQ9y3ltW5?usp=sharing)

### Code Execution
For train task, just execute
```python
pip install -r requirments.txt
python main.py --method dualcl --dataset amazon-jewel-balance
```


![image](output.png)

### Data Example：
| Text | Label |
|------|-------|
| The gold is not as shiny or colorful as advertised. However, I LOVE this purse. It's stiff which is great because when I put it in my car, it holds its shape. It has a compartment for an iPad, cell phone, pens, and an inside zippered pocket as well as an outside zippered pocket. It then also has a zippered middle compartment (big enough to put in folders/notebooks) and open compartments on either side of the zippered so you can easily separate items. Then there is a heavy-duty zipper to close it all up! It's a great purse that is stylish and just unique enough without being crazy. I LOVE LOVE LOVE it!!! | positive |
| I got these jeggings for a few costumes I hope to wear for comic-con. Looking more like Halloween. Anyways love the colors-just what I was searching for. However, I am a big girl with a few rolls-blush. The jeggings are snug couldn't get them past my belly button-just fell under it. The fit reminds me of 70's hip huggers that came back in the 90's-hence the thong/tramp stamp display. While these pants are straight leggers-80's-my decade, the waistband is 70's/90's hip huggers. My boyfriend reckoned they are meant to be that way; I agree. Don't try to pull them any higher or they will rip in the waistband-one pair did a little. They don't tell you that in the description. Also, they are something you have to wiggle in a few times to get them up on you. Zipper & button do work-when you lie down to zip up. You will have to peel them off a few times to get them used to your body. The price is nice, but the package was ripped when I got it-lucky all of the pants were accounted for. I am on the fence whether I want to send them back or not. If I will need a belt or not. Came in a reasonable amount of time. Hope this helps | positive |
| I bought these shoes as I had just purchased a pair of orthotics from the same company. I wanted a pair of clogs that would fit my foot's needs and had been looking for memory foam ones. However, upon arrival, I tried them on and was very disappointed. First off, the clogs themselves run a bit small and my foot was hanging out a bit. Second, I would not classify these as "memory foam" as it's an insert you can take out and it is very thin. Lastly, the "shell" of the clog is very flimsy and I believe a normal clog's shell should be much firmer. | negative |
| First of all, the color is not like what's in the picture. In the picture, the wallet looks like a moss green but oohhhhh noooo its a bright green almost like fluorescent green. Maybe if the outside was all black with the snakeskin and green on the inside I'll be happy but nah uh its yucky. Pfft idk what others really reacted to 1st seeing the wallet but from these reviews, I think they lying. ~unsatisfied customer~ | negative |




### Result
Here's the data formatted as a markdown table:

| Model | Data Condition | Data Num | Loss Type | Iteration | Split | Loss | Accuracy (%) | F1 Score |
|-------|----------------|----------|-----------|-----------|-------|------|--------------|----------|
| CE Loss | Balanced | 100,000 | CE Loss | 20,000 | Val | 0.2158 | 92.00 | 0.92 |
| CE Loss | Balanced | 100,000 | CE Loss | 20,000 | Test | 0.2180 | 91.00 | 0.91 |
| Dual Loss | Balanced | 100,000 | Dual Loss | 60,000 | Val | 1.9773 | 91.00 | 0.91 |
| Dual Loss | Balanced | 100,000 | Dual Loss | 60,000 | Test | 1.9792 | 90.00 | 0.90 |
| CE Loss | Unbalanced | 200,000 | CE Loss | - | Test | 0.1896 | 93.07 | 0.89 |
| CE Loss | Unbalanced | 200,000 | CE Loss | - | Val | 0.1870 | 93.05 | 0.89 |
| Dual CE Loss | Unbalanced | 200,000 | Dual CE Loss | - | Test | 2.0230 | 92.12 | 0.88 |
| Dual CE Loss | Unbalanced | 200,000 | Dual CE Loss | - | Val | 2.0203 | 92.11 | 0.88 |
| CE Loss | Unbalanced Small Num | 3,000 | CE Loss | - | Test | 0.9110 | 89.00 | 0.85 |
| CE Loss | Unbalanced Small Num | 3,000 | CE Loss | - | Val | 0.8199 | 90.20 | 0.83 |
| Dual Loss | Unbalanced Small Num | 3,000 | Dual Loss | - | Test | 3.5337 | 88.50 | 0.83 |
| Dual Loss | Unbalanced Small Num | 3,000 | Dual Loss | - | Val | 3.5768 | 88.30 | 0.83 |


