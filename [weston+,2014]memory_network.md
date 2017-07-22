#[Weston+,2014]Memory Network

A baseline model of the task.

##Memory Network

Four components:I,G,O and R.

I: Preprocessing, e.g. parsing, coreference and entity resolution for text inputs. Encode the input into an internal feature representation, e.g. embedding.

G:Storing memory in a slot. Updating old memories given the new input.

O:Produces a new output in feature representation space. 

R:Converts the output into the response format desired. e.g. decoding.

##For Text

The core of the inference lies in the O and R modules.

O module chooses the highest (or top 2) scored supporting memory and the R module ranks the words and produce a textual response r.

<a href="http://www.codecogs.com/eqnedit.php?latex=o_{1}=O_{1}(x,m)=\underset{i=1,...,N}{argmax}&space;s_{O}(x,m_i)" target="_blank"><img src="http://latex.codecogs.com/gif.latex?o_{1}=O_{1}(x,m)=\underset{i=1,...,N}{argmax}&space;s_{O}(x,m_i)" title="o_{1}=O_{1}(x,m)=\underset{i=1,...,N}{argmax} s_{O}(x,m_i)" /></a>

<a href="http://www.codecogs.com/eqnedit.php?latex=o_{2}=O_{2}(x,m)=\underset{i=1,...,N}{argmax}&space;s_{O}([x,m_{O_1}],m_i)" target="_blank"><img src="http://latex.codecogs.com/gif.latex?o_{2}=O_{2}(x,m)=\underset{i=1,...,N}{argmax}&space;s_{O}([x,m_{O_1}],m_i)" title="o_{2}=O_{2}(x,m)=\underset{i=1,...,N}{argmax} s_{O}([x,m_{O_1}],m_i)" /></a>

<a href="http://www.codecogs.com/eqnedit.php?latex=r=\underset{w\in&space;W}{argmax}s_R([x,m_{O_1},m_{O_2}],w)" target="_blank"><img src="http://latex.codecogs.com/gif.latex?r=\underset{w\in&space;W}{argmax}s_R([x,m_{O_1},m_{O_2}],w)" title="r=\underset{w\in W}{argmax}s_R([x,m_{O_1},m_{O_2}],w)" /></a>

Scoring functions sO and sR have the same form that of an embedding model:

<a href="http://www.codecogs.com/eqnedit.php?latex=s(x,y)={\Phi_{x}(x)}^{\top&space;}U^{\top}U{\Phi}_{y}(y)" target="_blank"><img src="http://latex.codecogs.com/gif.latex?s(x,y)={\Phi_{x}(x)}^{\top&space;}U^{\top}U{\Phi}_{y}(y)" title="s(x,y)={\Phi_{x}(x)}^{\top }U^{\top}U{\Phi}_{y}(y)" /></a>

Length of sO is 3|W|, one for <a href="http://www.codecogs.com/eqnedit.php?latex=\Phi_y(.)" target="_blank"><img src="http://latex.codecogs.com/gif.latex?\Phi_y(.)" title="\Phi_y(.)" /></a> and two for <a href="http://www.codecogs.com/eqnedit.php?latex=\Phi_y(.)" target="_blank"><img src="http://latex.codecogs.com/gif.latex?\Phi_x(.)" title="\Phi_y(.)" /></a>.

Minimize over model paramenters U<sub>O</sub> and U<sub>R</sub>.

<a href="http://www.codecogs.com/eqnedit.php?latex=\sum_{\bar{f}\neq&space;m_{O_1}}&space;max(0,\gamma&space;-s_O(x,m_{O_1})&plus;s_O(x,\bar{f}))&space;&plus;&space;\sum_{{\bar{f}}'\neq&space;m_{O_2}}&space;max(0,\gamma&space;-s_O([x,m_{O_1}],m_{O_2})&plus;s_O([x,m_{O_1}],{\bar{f}}'))&space;&plus;&space;\sum_{\bar{r}\neq&space;r}&space;max(0,\gamma&space;-s_O([x,m_{O_1},m_{O_2}],r)&plus;s_O([x,m_{O_1},m_{O_2}],\bar{r}))" target="_blank"><img src="http://latex.codecogs.com/gif.latex?\sum_{\bar{f}\neq&space;m_{O_1}}&space;max(0,\gamma&space;-s_O(x,m_{O_1})&plus;s_O(x,\bar{f}))&space;&plus;&space;\sum_{{\bar{f}}'\neq&space;m_{O_2}}&space;max(0,\gamma&space;-s_O([x,m_{O_1}],m_{O_2})&plus;s_O([x,m_{O_1}],{\bar{f}}'))&space;&plus;&space;\sum_{\bar{r}\neq&space;r}&space;max(0,\gamma&space;-s_O([x,m_{O_1},m_{O_2}],r)&plus;s_O([x,m_{O_1},m_{O_2}],\bar{r}))" title="\sum_{\bar{f}\neq m_{O_1}} max(0,\gamma -s_O(x,m_{O_1})+s_O(x,\bar{f})) + \sum_{{\bar{f}}'\neq m_{O_2}} max(0,\gamma -s_O([x,m_{O_1}],m_{O_2})+s_O([x,m_{O_1}],{\bar{f}}')) + \sum_{\bar{r}\neq r} max(0,\gamma -s_O([x,m_{O_1},m_{O_2}],r)+s_O([x,m_{O_1},m_{O_2}],\bar{r}))" /></a>

Training uses the margin loss function and SGD for opt. Randomly sample <a href="http://www.codecogs.com/eqnedit.php?latex=\Phi_y(.)" target="_blank"><img src="http://latex.codecogs.com/gif.latex?\bar{f}" title="\Phi_y(.)" /></a> and so on rather than compute the whole sum for each training example.

The R module is often replaced by RNN or LSTM. Here uses a single word response as above.

##Extendsion

###Write time

To model the write in time of the memory, add three more new features which takes on 0 or 1, to represent which memory is earlier put in. Meanwhile change the scoring of the O module. Instead of scoring input sentences, candidate memory pairs with s as above learn a function on triples.

<a href="http://www.codecogs.com/eqnedit.php?latex=s_{O_t}(x,y,y')=\Phi_x(x)^{\top}U_{O_t}^{\top}U_{O_t}(\Phi_y(y)-\Phi_y(y')&plus;\Phi(x,y,y'))" target="_blank"><img src="http://latex.codecogs.com/gif.latex?s_{O_t}(x,y,y')=\Phi_x(x)^{\top}U_{O_t}^{\top}U_{O_t}(\Phi_y(y)-\Phi_y(y')&plus;\Phi(x,y,y'))" title="s_{O_t}(x,y,y')=\Phi_x(x)^{\top}U_{O_t}^{\top}U_{O_t}(\Phi_y(y)-\Phi_y(y')+\Phi(x,y,y'))" /></a>

###Unseen words

For unseen words, they extend the features to 8|W| by adding the bag of words matching score to the learned embedding score with a mixing parameter. Unseen words can be modeled similarly by using matching features on their context words.

##Experiments

Large-scale QA and bAbI simulated world QA

Reached all 100% on the simulation QA task. bAbI is only a testing data.

Reached 0.82 F1 on the Large-Scale QA.

Performed well on unseen words. And meanwhile can give a simple grammar for generating true answers in sentence form using LSTM.

##Future Works

Combine models effectively to answer both general knowledge questions and specific statements relating to the previous dialogue.

On harder QA and open-domain machine comprehension tasks. 

Coreference, sentences more stuctured and requiring more temporal and causal understanding. Sophisticated architectures and memory management, sentence representations. Weakly supervised settings should be concerned too.

##Personal Issues

感觉是非常值得挖掘的模型，但是拿这个当baseline做优化的话有一点重复造轮子的感觉，但是这篇论文本身是非常Golden的，主要是想法非常好。

Future Works里面大部分内容已经被实现了，要做进一步的研究的话以更加Sophisticated的模型开始比较好。

做了一下实现，效果还是不错的。