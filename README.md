# detect-malicious-URLs
项目并无多少实际效用，纯粹做来玩。针对恶意url做了ML和DL两个特征提取，分别使用了各种机器学习模型和基于Attention的LSTM。

<br><hr>
其中dl里
<pre>
运行train.py用于训练，<br>
运行predict.py用于预测
做了相关比对试验(RNN, GRU, 单LSTM, 双向LSTM)
</pre>
<br>
ml里
<pre>提取了Tfidf+Hashing特征，使用LinearSVC</pre>
