# Rice-leaf-disease-prediction-using-Convolutional-Neural-Network
Create a Machine learning model that can classify images of the three major attacking diseases of Rice plants they are Bacterial blight, Brown spot, and Leaf smut.

### Bacterial Blight:- 
It is a deadly bacterial disease that is among the most destructive afflictions of cultivated rice (Oryza sativa and O. glaberrima). The disease was first observed in 1884–85 in Kyushu, Japan, and the causal agent, the bacterium Xanthomonas oryzae pathovar oryzae (also referred to as Xoo), was identified in 1911, at that time having been named Bacillus oryzae. Thriving in warm, humid environments, bacterial blight has been observed in rice-growing regions of Asia, the western coast of Africa, Australia, Latin America, and the Caribbean.

Bacterial blight first becomes evident as water-soaked streaks that spread from the leaf tips and margins, becoming larger and eventually releasing a milky ooze that dries into yellow droplets. Characteristic grayish white lesions then appear on the leaves, signaling the late stages of infection, when leaves dry out and die. In seedlings, the leaves dry out and wilt, a syndrome known as kresek. Infected seedlings usually are killed by bacterial blight within two to three weeks of being infected; adult plants may survive, though rice yield and quality are diminished.

Since rice paddies are flooded throughout most of the growing season, Xoo may easily spread among crops; bacteria travel through the water from infected plants to the roots and leaves of neighboring rice plants. Wind and water may also help spread Xoo bacteria to other crops and rice paddies. Various mechanisms of disease, including quorum sensing and biofilm formation, have been observed in rice bacterial blight and Xoo. In addition to rice, Xoo may infect other plants, such as rice cut-grass (Leersia oryzoides), Chinese sprangletop (Leptochloa chinensis), and common grasses and weeds. In nongrowing seasons, Xoo may survive in rice seeds, straw, other living hosts, water, or, for brief periods, soil.

### Brown Spot:- 
The brown spot has been historically largely ignored as one of the most common and most damaging rice diseases.

It is a fungal disease that infects the coleoptile, leaves, leaf sheath, panicle branches, glumes, and spikelets. Its most observable damage is the numerous big spots on the leaves which can kill the whole leaf. When infection occurs in the seed, unfilled grains or spotted or discolored seeds are formed.

The disease can develop in areas with high relative humidity (86−100%) and temperature between 16 and 36°C. It is common in unflooded and nutrient-deficient soil, or in soils that accumulate toxic substances.

For infection to occur, the leaves must be wet for 8−24 hours.

The fungus can survive in the seed for more than four years and can spread from plant to plant through air. Major sources of brown spots in the field include

1. Infected seed, which gives rise to infected seedlings.
2. Volunteer rice.
3. Infected rice debris.
4. Weeds.

Brown spots can occur at all crop stages, but the infection is most critical during maximum tillering up to the ripening stages of the crop. It also causes both quantity and quality losses.

On average, the disease causes a 5% yield loss across all lowland rice production in South and Southeast Asia. Severely infected fields can have as high as 45% yield loss.

Heavily infected seeds cause seedling blight and lead to 10−58% seedling mortality. It also affects the quality and the number of grains per panicle and reduces the kernel weight.

In terms of history, the Brown Spot was considered to be the major factor contributing to the Great Bengal Famine of 1943.

### Leaf Smut:- 
Leaf smut in rice, caused by the fungus Entyloma oryzae, is a disease that primarily affects the leaves of rice plants, resulting in small, black smut sori, or fungal fruiting bodies, appearing as scattered or aggregated black spots. These infected areas around the sori often turn yellow, reducing the leaf's overall green area, which in turn diminishes the plant's photosynthetic capacity and can lead to premature leaf death.

The disease thrives in high humidity, warm temperatures, and densely planted fields with poor air circulation. The fungus enters the plant through natural openings or wounds, producing spores that spread to other plants via wind, water, or mechanical means.

Effective management of leaf smut includes planting resistant rice varieties, implementing proper cultural practices such as balanced fertilization and adequate spacing, and using fungicides judiciously as part of an integrated pest management approach. While the economic impact of leaf smut is generally less severe compared to other rice diseases, it can still cause significant yield losses if not properly managed, particularly in susceptible rice varieties and under favorable conditions for the fungus.


### Overall Analysis
Each disease has distinct symptoms and impacts on rice plants, affecting their growth, photosynthesis, and overall yield. Effective disease management strategies include using disease-resistant rice varieties, implementing proper field hygiene practices, and avoiding conditions that promote disease spread. Early detection and prompt intervention are crucial to minimizing the economic and agronomic losses caused by these diseases. The classification model developed in this project will contribute to early disease detection and timely management, helping to ensure healthier rice crops and higher yields. Remember that accurate disease diagnosis requires the expertise of plant pathologists. This analysis serves as a general overview, and specific diagnosis should involve consultation with experts in the field.

## Results from the Analysis
- We started our model building by custom convolutional neural network from scratch through **Tensorflow-keras module**, but as our dataset was very small, getting an acceptable and satisfactory accuracy score was difficult through our normal CNN models. Tuning various hyperparameters and using different custom CNN architectures didn't yield us a good result.
- **Transfer learning** was carried out following the failure of **custom CNN** models. these models are pre-trained on large datasets such as **Imagenet**. Knowledge gained from huge datasets helps the transfer-learning models to give us good results for our new dataset with very few data samples.
- **Resnet50**, **InceptionV3**, **Xception**, and **EfficientNet** Transfer learning models were used for our project.
- The **Xception** model performed better compared to the other transfer learning models. It gave us an accuracy score of **0.997 (99.7 %)** for the Non-augmented training sets and a validation accuracy of **0.4 (40 %)** for the Non-augmented test set.
- When the **Xception** model was subjected to an Augmented training dataset it got an accuracy score of **0.997 (99.7 %)** and a validation accuracy score of **0.7 (70 %)** for the test set (Non-augmented) which was highest of any models trained so far in our dataset. 
- But still, we can see that overfitting is in action throughout our model, learning noise rather than the underlying pattern in our dataset. This will be eliminated if we introduce more training samples for our model to learn from and generalize better to unseen data.
