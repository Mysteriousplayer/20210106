# [TMM 2022] Micro-Influencer Recommendation by Multi-Perspective Account Representation Learning

## News
A new version of the paper has been published.

## Abstract

<div align=center>
<img src="https://github.com/Mysteriousplayer/TMM22-MORNING/blob/main/fig4.png" width="50%">
</div>

> Influencer marketing is emerging as a new marketing method, changing the marketing strategies of brands profoundly. In order to help brands find suitable micro-influencers as marketing partners, the micro-influencer recommendation is regarded as an indispensable part of influencer marketing. However, previous works only focus on modeling the individual image of brands/micro-influencers, which is insufficient to represent the characteristics of brands/micro-influencers over the marketing scenarios. In this case, we propose a micro-influencer ranking joint learning framework which models brands/micro-influencers from the perspective of individual image, target audiences, and cooperation preferences. Specifically, to model accounts’ individual image, we extract topics information and images semantic information from historical content information, and fuse them to learn the account content representation. We introduce target audiences as a new kind of marketing role in the micro-influencer recommendation, in which audiences information of brand/micro-influencer is leveraged to learn the multi-modal account audiences representation. Afterward, we build the attribute co-occurrence graph network to mine cooperation preferences from social media interaction information. Based on account attributes, the cooperation preferences between brands and micro-influencers are refined to attributes’ co-occurrence information. The attribute node embeddings learned in the attribute co-occurrence graph network are further utilized to construct the account attribute representation. Finally, the global ranking function is designed to generate ranking scores for all brand-micro-influencer pairs from the three perspectives jointly. The extensive experiments on a publicly available dataset demonstrate the effectiveness of our proposed model over the state-of-the-art methods. 

## Framework

<div align=center>
<img src="https://github.com/Mysteriousplayer/TMM22-MORNING/blob/main/fig5.png">
</div>

<div align=center>
<img src="https://github.com/Mysteriousplayer/TMM22-MORNING/blob/main/fig6.png">
</div>

-We propose the MORNING by novelly modeling social media accounts from the perspective of individual image, target audiences, and cooperation preferences. Experimental results show that our method achieves state-of-the-art performance.
-We introduce target audiences as a new kind of marketing role in the micro-influencer recommendation. Multi-modal audiences information is utilized to learn the account audiences representation. And we collect a social-media-audiences dataset, which can benefit future research.
-We successfully mine cooperation preferences from interaction information on social media. Based on account attributes, we build the attribute co-occurrence graph network to capture cooperation preferences at the attribute level, and the attribute node embeddings are further utilized to construct the account attribute representation.

## Installation
Install all requirements required to run the code on a Python 3.6.13 by:
> First, you need activate a new conda environment.
> 
> pip install -r requirements.txt

## Datasets
Brand-micro-influencer dataset: plase contact email: gantian@sdu.edu.cn.

features link: [https://pan.baidu.com/s/1NML0tH8Z50_LkO-kdPxslQ](https://pan.baidu.com/s/1IgpCWdWxY2BUuPrUp-krtA ) 
code：2009

## Training

```
bash run_morning.sh
```



## Cite
If you found our work useful for your research, please cite our work:
```
 @ARTICLE{wang_2023, 
author={Wang, Shaokun and Gan, Tian and Liu, Yuan and Wu, Jianlong and Cheng, Yuan and Nie, Liqiang}, 
journal={IEEE Transactions on Multimedia}, 
title={Micro-Influencer Recommendation by Multi-Perspective Account Representation Learning}, 
year={2023}, 
volume={25}, 
pages={2749-2760}}
```
