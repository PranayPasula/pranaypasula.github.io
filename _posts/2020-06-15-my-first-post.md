---
layout: post
title: Welcome!
---

<div class="container-fluid">
<div class="row">
<div class="col-4" style="float:left;">
<img src="/assets/pic.jpg" class="img-fluid">
</div>
<div class="col" markdown="1">
I'm a recent graduate of Berkeley EECS, where I was fortunate to be advised by <a href="https://people.eecs.berkeley.edu/~dawnsong/">Dawn Song</a> and <a href="https://ruoxijia.info/">Ruoxi Jia</a>. 

Before Berkeley, I spent 3 years in industry, where I independently launched deep learning and classical machine learning programs to capture high-value opportunities.

My research interests are primarily in deep learning, reinforcement learning (RL), and unsupervised learning. I also have experience in computer vision, robotics, optimization, and natural language processing.

I enjoy developing machine learning algorithms as well as software and systems that accelerate these algorithms. Outside of work, I like biking, playing tennis, and hanging out with cats (though dogs are a very close second).

<div style="font-weight:500"><a href="https://linkedin.com/in/pranaypasula">LinkedIn</a> &nbsp; | &nbsp; <a href="https://scholar.google.com/citations?user=QMPZJQUAAAAJ&hl=en">Google Scholar</a> &nbsp; | &nbsp; Email: <a href="mailto:pasula@berkeley.edu">pasula@berkeley.edu</a></div>
</div>
</div>
</div>

<h1 style="margin-top:3.0em;margin-bottom:-0.5em;font-weight:450">Education</h1>
<hr>

<div class="row" style="margin-bottom:1.0em">
<div class="col-2 col-lg-1">
<img src="/assets/cal_seal.png" class="img-fluid" style="max-width:50px;max-height:50px;width:100%;display:block;margin-left:auto;margin-right:auto">
</div>

<div class="col-10 col-lg-5 px-0" style="margin-bottom:1.0em" markdown="1">
<h5 style="margin-top:-0.33em;margin-bottom:0.0em;font-size:18px">
University of California, Berkeley
</h5>
Master's in Electrical Engineering and Computer Science

GRE: 170/170 Quant, 169/170 Verbal (99.9th percentile)

Selected coursework
- Advanced Robotics w/ Pieter Abbeel<a href="#phd" style="color:blue">*</a>
- Deep Reinforcement Learning w/ Sergey Levine<a href="#phd" style="color:blue">*</a>
- Statistical Learning Theory w/ Ben Recht and Moritz Hardt<a href="#phd" style="color:blue">*</a>
- Deep Unsupervised Learning w/ Pieter Abbeel<a href="#phd" style="color:blue">*</a><a href="#audit" style="color:blue">&#8224;</a> 
- Optimization Analysis and Algorithms w/ Martin Wainwright<a href="#phd" style="color:blue">*</a>
- Algorithmic Human-Robot Interaction w/ Anca Dragan<a href="#phd" style="color:blue">*</a>
<!-- - R&D Tech Management and Ethics w/ Lee Fleming<a href="#master" style="color:blue">**</a> -->
</div>

<div class="col-2 col-lg-1">
<img src="/assets/uh_symbol.png" class="img-fluid" style="max-width:45px;max-height:45px;width:100%;display:block;margin-left:auto;margin-right:auto">
</div>

<div class="col-10 col-lg-5 px-0" markdown="1">
<h5 style="margin-top:-0.33em;margin-bottom:0.0em;font-size:18px">
University of Houston 
</h5>
Bachelor's in Electrical Engineering, *summa cum laude*

Selected coursework
- Computer Vision<a href="#phd" style="color:blue">*</a>
- Machine Learning<a href="#phd" style="color:blue">*</a>
- Digital Image Processing<a href="#phd" style="color:blue">*</a>
- Advanced Algorithms
- Operating Systems
- Computer Architecture
- Embedded Systems
</div>
</div>

<div class="row" style="margin-top:-1.0em">
<div class="col">
<table cellpadding="1px">
<tr>
<td valign="top" style="border:none;text-align:right">
<span style="color:blue">*</span>
<br>
<!-- <span style="color:blue">**</span> 
<br> -->
<span style="color:blue">&#8224;</span>
</td>

<td valign="top" style="float:left;border:none;color:#a0aec0">
<a id="phd">PhD-level course</a>
<br>
<!-- <a id="master">Master-level course</a>
<br> -->
<a id="audit">Audited to support research interests</a>
</td>
</tr>

</table>
</div>
</div>

<h1 style="margin-bottom:-1.0em;margin-bottom:-0.5em;font-weight:400">Experience</h1>
<hr>

<h2 style="margin-bottom:1.0em;font-weight:400;font-size:26px">Academic</h2>

<div class="row">
<div class="col-2">
<div style="float:right;margin:0px">
<img src="/assets/run.gif" class="img-fluid" style="margin:0px;margin-right:-1px;display:inline;float:left">
<img src="/assets/jump.gif" class="img-fluid" style="margin:0px;display:inline;">
</div>
<div style="float:right">
<img src="/assets/cartwheel.gif" class="img-fluid" style="margin:0px;margin-right:-1px;display:inline;float:left">
<img src="/assets/aerial.gif" class="img-fluid" style="margin:0px;display:inline;">
</div>
<span style="font-size:9px;margin-top:4px;text-align:right;float:right;clear:both">Images adapted from Xue Bin Peng's <a href="https://bair.berkeley.edu/blog/2018/04/10/virtual-stuntman/">BAIR blog post</a></span>
</div>

<div class="col" markdown="1">
<h5 style="margin-top:-0.33em;margin-bottom:0.0em;font-size:18px">
<a id="complex">
Complex Skill Acquisition through Simple Skill Imitation Learning
</a>
</h5>
<p style="font-size:14px;margin-top:0.0em;margin-bottom:1.0em;color:gray">Author: Me</p>
<a href="https://arxiv.org/abs/2007.10281" class="btn btn-primary btn-sm">Paper</a>

<strong>Topics:</strong> reinforcement learning, deep learning, unsupervised learning

Humans have the power to reason about complex tasks as combinations of simpler, interpretable subtasks. There are many hierarchical reinforcement learning approaches designed to handle tasks comprised of <em>sequential</em> subtasks, but what if a task is made up of <em>concurrent</em> subtasks?

We propose a novel objective function that regularizes a version of the VAE objective in order to induce latent space structure that captures the relationship between a behavior and the subskills that comprise this behavior in a <em>disentangled</em> and <em>interpretable</em> way.  We evaluate both the original and new objectives on a moderately complex imitation learning problem from the <a href="https://xbpeng.github.io/projects/DeepMimic/index.html">DeepMimic</a> library, in which agents are trained to perform a behavior after being trained on subskills that qualitatively comprise that behavior.

<strong>Keywords: </strong> hierarchical imitation learning, adversarial imitation learning, generative models, autoregressive models, mutual information maximization
</div>
</div>
<hr>

<div class="row">
<div class="col-2" style="float:right">
<img src="/assets/prox1.gif" class="img-fluid" style="margin:0.0px;">
<img src="/assets/prox2.gif" class="img-fluid" style="margin:0.0px;">
<span style="font-size:9px;margin-top:4px;float:right">Images adapted from <a href="https://pierreablin.com/">Pierre Ablin</a></span>
</div>

<div class="col" markdown="1">
<h5 style="margin-top:-0.33em;margin-bottom:0.0em;font-size:18px">
<a id="lagrangian">
Lagrangian Duality in Reinforcement Learning
</a>
</h5>
<p style="font-size:14px;margin-top:0.0em;margin-bottom:1.0em;color:gray">Author: Me</p>
<a href="https://arxiv.org/abs/2007.09998" class="btn btn-primary btn-sm">Paper</a>

<strong>Topics:</strong> reinforcement learning, optimization

Though duality is used extensively in certain fields, such as supervised learning in machine learning, it has been much less explored in others, such as reinforcement learning. In this paper, we show how duality is involved in a variety of RL work, from that which spearheaded the field (e.g. Richard Bellman's <a href="https://en.wikipedia.org/wiki/Bellman_equation">tabular value iteration</a>) to that which has been put forth within just the past few years yet has already had significant impact (e.g. <a href="https://arxiv.org/abs/1502.05477">TRPO</a>, <a href="https://arxiv.org/abs/1602.01783">A3C</a>, <a href="https://arxiv.org/abs/1606.03476">GAIL</a>).

We show that duality is not uncommon in reinforcement learning, especially when <em>value iteration</em>, or <em>dynamic programming</em>, is used or when first or second order approximations are made to transform initially intractable problems into tractable convex programs.

In some cases duality is used as a theoretical tool to prove certain results or to gain insight into the meaning of the problem involved. In other cases duality is leveraged to employ gradient-based methods over some <em>dual</em> space, as is done in <a href="https://web.stanford.edu/~boyd/papers/admm_distr_stats.html">alternating direction method of multipliers (ADMM)</a>, <a href="https://web.iem.technion.ac.il/images/user-files/becka/papers/3.pdf">mirror descent</a>, and <a href="https://stanford.edu/~jduchi/projects/DuchiAgWa12.pdf">dual averaging</a>.

<strong>Keywords: </strong> duality, optimization, mirror descent, dual averaging, alternating method of multipliers (ADMM), value iteration, trust region policy optimization (TRPO), asynchronous actor critic (A3C), generative adversarial imitation learning (GAIL)
</div>
</div>
<hr>

<div class="row">
<div class="col-2">
<img src="/assets/pacman_smaller.gif" class="img-fluid" style="float:right;">
</div>

<div class="col" markdown="1">
<h5 style="margin-top:-0.33em;margin-bottom:0.0em;font-size:18px">
<a id="negative">
Negative Interference in Multi-Task Reinforcement Learning due to Pixel Observation Space Differences
</a>
</h5>
<p style="font-size:14px;margin-top:0.0em;margin-bottom:1.0em;color:gray">Author: Me | Thanks to Pieter Abbeel for proposing the base experiment design!</p>
<a href="https://pranaypasula.github.io/multitask_rl_visual_interference.html" class="btn btn-primary btn-sm">Results</a>
<a href="https://pranaypasula.github.io/data_visualization_neg_interference_exp_tasks.html" class="btn btn-primary btn-sm">Data Visualization</a>
<a href="https://github.com/pranaypasula/mtrl-visual-interference" class="btn btn-primary btn-sm">Code</a>

<strong>Topics:</strong> reinforcement learning, deep learning, distributed systems, parallel computing

Transfer learning has revolutionized certain fields, such as computer vision and natural language processing, but reinforcement learning has not yet seen similar benefit. One reason is that RL tasks are generally more specific than images or sentences are. Namely, some ways that tasks differ are in states, actions, reward schemes, and temporal horizons.

To better understand how different raw pixel state distributions affect multi-task reinforcement learning, we evaluate multi-task policies on a set of tasks that differ only in state distribution. In this work we focus on how the DQN architecture and some of its variants handle multi-task learning of Atari games that differ only in raw pixel observation space.

<strong>Keywords: </strong>multi-task reinforcement learning, transfer learning, generalization, DQN, parallel computing
</div>
</div>
<hr>

<div class="row">
<div class="col-2">
<img src="/assets/deepfake_thumb.jpg" class="img-fluid" style="float:right">
</div>

<div class="col" markdown="1">
<h5 style="margin-top:-0.33em;margin-bottom:0.0em;font-size:18px">
<a id="robust">
Robust Detection of Manipulated Videos (i.e. Deepfakes)
</a>
</h5>
<p style="font-size:14px;margin-top:0.0em;margin-bottom:1.0em;color:gray">Authors: Me, Ujjwal Singhania, Yijiu Zhong | Thanks to Dawn Song and Ruoxi Jia for advising this work!</p>
<a href="/assets/deepfake_poster.pdf" class="btn btn-primary btn-sm">Poster</a>
<a href="/assets/deepfake_slides.pdf" class="btn btn-primary btn-sm">Slides</a>
<a href="mailto:pasula@berkeley.edu?subject=Code Request - Deepfake Detection&body=Please add reason for request here" class="btn btn-primary btn-sm">Request Code</a>

<strong>Topics:</strong> computer vision, deep learning, image processing

Artificial videos are becoming indistinguishable from real content. For example, <a href="https://www.theverge.com/tldr/2018/4/17/17247334/ai-fake-news-video-barack-obama-jordan-peele-buzzfeed">videos of world leaders</a> have been manipulated to make them appear to say incendiary things. Previous approaches have failed to generalize over varied deepfake generation methods. We use machine learning and computer vision to create a classifier that detects deepfake videos and is robust to unseen manipulation methods.

<strong>Keywords: </strong> deepfake detection, robustness, generalization, <a href="https://www.kaggle.com/c/deepfake-detection-challenge">Deepfake Detection Challenge (DFDC)</a>, optical flow, <a href="https://arxiv.org/abs/1912.13458">Face X-ray</a>, <a href="https://arxiv.org/abs/1905.11946">EfficientNet</a>
</div>
</div>
<hr>

<div class="row">
<div class="col-2">
<img src="/assets/opt_resized.gif" class="img-fluid" style="margin:0.0px;float:right">
<span style="font-size:9px;margin-top:4px;float:right">Image adapted from <a href="https://twitter.com/alecrad">Alec Radford</a></span>
</div>

<div class="col" markdown="1">
<h5 style="margin-top:-0.33em;margin-bottom:0.0em;font-size:18px">
<a id="lipgd">
LipGD: Faster Gradient Descent through Efficient Lipschitz Constant Approximations
</a>
</h5>
<p style="font-size:14px;margin-top:0.0em;margin-bottom:1.0em;color:gray">Author: Me</p>
<a href="mailto:pasula@berkeley.edu?subject=Code Request - LipGD&body=Please add reason for request here" class="btn btn-primary btn-sm">Request Code</a>

<strong>Topics:</strong> optimization, deep learning

Optimization theory gives us nice guarantees on the optimality and convergence rates of convex problems. A quantity of particular interest is the <a href="https://en.wikipedia.org/wiki/Lipschitz_continuity">Lipschitz constant</a> of the derivative of the function being optimized. The reciprocal of this quantity upper bounds the step size of the gradient update.

Unfortunately, most problems of interest are highly non-convex, so such guarantees don’t directly apply. However, we see empirically that there generally exists a neighborhood around any given iterate of parameter values that is convex, and such guarantees do directly apply within this neighborhood. We also see empirically that the Lipschitzness of the gradient doesn’t change much from iterate to iterate. We use this information to safely take gradient steps that are larger than those dictated by common step size schedules, leading to faster convergence.

Certain problems, such as those involving adversarial training (e.g. <a href="https://arxiv.org/abs/1701.07875">Wasserstein GAN</a>), advocate <em>momentum-free</em> optimization, which has <em>much</em> slower convergence than <em>momentum-based</em> optimization. We believe that our approach could be especially useful in addressing these problems.

<strong>Keywords: </strong> Lipschitz, smoothness, stochastic gradient descent, non-convex optimization, momentum-free optimization, Wasserstein GAN (WGAN)
</div>
</div>
<hr>

<div class="row">
<div class="col-2">
<img src="/assets/space.gif" class="img-fluid" style="float:right">
</div>

<div class="col" markdown="1">
<h5 style="margin-top:-0.33em;margin-bottom:0.0em;font-size:18px">
<a id="improving">
Improving Multi-Task Reinforcement Learning through Disentangled Representation Learning
</a>
</h5>
<p style="font-size:14px;margin-top:0.0em;margin-bottom:1.0em;color:gray">Author: Me</p>
<a href="/assets/pasula2019improving.pdf" class="btn btn-primary btn-sm">Paper</a>

<strong>Topics:</strong> reinforcement learning, unsupervised learning, deep learning

When humans learn to perform a task, they tend to also improve their skills on related tasks, even without explicitly practicing these other tasks. In reinforcement learning (RL), the multi-task setting aims to leverage similarities across tasks to help agents more quickly learn multiple tasks simultaneously. However, multitask RL has a number of key issues, such as negative interference, that make it difficult to implement in practice.

We propose an approach that learns disentangled representations in order to alleviate these issues and find effective multi-task policies in a high-dimensional raw-pixel observation space. We show that this approach can be superior to other multi-task RL techniques with little additional cost. We also investigate disentanglement itself by capturing, adjusting, and reconstructing latent representations that have been learned from Atari images and gain insight into their underlying meaning.

<strong>Keywords: </strong> multi-task reinforcement learning, disentanglement, representation learning, variational autoencoder, DQN
</div>
</div>
<hr>

<div class="row">
<div class="col-2">
<img src="/assets/battlezone.gif" class="img-fluid" style="float:right">
</div>

<div class="col" markdown="1">
<h5 style="margin-top:-0.33em;margin-bottom:0.0em;font-size:18px">
<a id="empirical">
Empirical Evaluation of Double DQN under Monotonic Reward Transformations
</a>
</h5>
<p style="font-size:14px;margin-top:0.0em;margin-bottom:1.0em;color:gray">Author: Me</p>
<a href="https://docs.google.com/document/d/1VO2uwpy1BHc8EtT0NzkhT5j6iCWvmIT7RcDG7a0Ucrc/edit?usp=sharing" class="btn btn-primary btn-sm">Paper</a>

<strong>Topics:</strong> reinforcement learning, deep learning

Several <a href="https://openai.com/blog/openai-baselines-dqn/">tricks</a> are employed to get DQNs to learn favorable policies. One of these is the use of reward clipping, in which rewards obtained by the agent are clipped to the interval [-1, +1] from their original value. Though some Atari games, such as Pong have only reward values that fall within this interval, most Atari games have no reward values within this interval. One stark example is Battlezone, in which the smallest reward and finest granularity of reward are both 1000.

For tasks with rewards primarily outside of [-1, +1], clipping rewards to this interval qualitatively changes the objective maximized by DQN to the <em>frequency of rewards</em> rather than the <em>quantity of overall reward</em>. In some cases these two correlate well, and performance may not be affected much by using clipped versus unclipped rewards. However, it’s easy to find tasks in which this invariance does not hold. In this work we propose three carefully chosen reward transformations and explore the performance of Double DQN under each of these as well as under the de facto reward clipping transformation.

<strong>Keywords: </strong> DQN, reward shaping
</div>
</div>
<hr>

<div class="row">
<div class="col-2">
<img src="/assets/drcwalk1-edit.gif" class="img-fluid" style="float:right;margin:0px">
<span style="font-size:9px;margin-top:4px;float:right">Image adapted from <a href="http://joschu.net/">John Schulman</a></span>
</div>

<div class="col" markdown="1">
<h5 style="margin-top:-0.33em;margin-bottom:0.0em;font-size:18px">
TrajOpt Presentation
</h5>
<p style="font-size:14px;margin-top:0.0em;margin-bottom:1.0em;color:gray">Author: Me, Eugene Vinitsky</p>
<a href="/assets/trajopt_presentation.pdf" class="btn btn-primary btn-sm">Slides</a>

<strong>Topics:</strong> robotics, optimization

<a href="http://rll.berkeley.edu/trajopt/ijrr/2013-IJRR-TRAJOPT.pdf">TrajOpt</a> is a trust region optimization-based motion planning algorithm that tends to produce high-quality trajectories more quickly and reliably than the leading motion planning algorithms at the time of its inception, such as <a href="https://www.ri.cmu.edu/pub_files/2013/5/CHOMP_IJRR.pdf">CHOMP</a>, do.

We give a 30 minute talk on TrajOpt, covering the state of motion planning at the time, the TrajOpt algorithm, its key ideas, a detailed comparison between TrajOpt and CHOMP, and its shortcomings. We also lead a discussion on building upon TrajOpt and on addressing its issues.

<strong>Keywords: </strong> trajectory optimization, motion planning, trust region method, collision checking
</div>
</div>
<hr>

<div class="row">
<div class="col-2">
<img src="/assets/beam.gif" class="img-fluid" style="float:right">
</div>

<div class="col" markdown="1">
<h5 style="margin-top:-0.33em;margin-bottom:0.0em;font-size:18px">
<a id="vae">
Variational Autoencoder (VAE) Viewer
</a>
</h5>
<p style="font-size:14px;margin-top:0.0em;margin-bottom:1.0em;color:gray">Author: Me</p>
<a href="https://github.com/pranaypasula/vae-viewer" class="btn btn-primary btn-sm">Code</a>

<strong>Topics:</strong> unsupervised learning, deep learning

A portable implementation of variational autoencoders (and some variants). This tool can be used to quickly view learned latent representations of user-provided images.

<strong>Keywords: </strong> (beta) variational autoencoder, representation learning
</div>
</div>
<hr>

<h2 style="margin-bottom:1.0em;font-weight:400;font-size:26px">Industry</h2>

<div class="row">
<div class="col-2 col-lg-1">
<img src="/assets/shell_symbol.png" class="img-fluid" style="max-width:50px;max-height:50px;width:100%;display:block;margin-left:auto;margin-right:auto">
</div>

<div class="col-10 col-lg-11" markdown="1">
<h3 style="margin-top:-0.33em;margin-bottom:1.0em;">
Shell Oil Company
</h3>
<h4 style="margin:0"> 
Deep Learning Research Engineer (Enhanced Problem Solving Engineer)
</h4>
<div style="color:#a0aec0;margin-bottom:1.0em">
2016 &ndash; 2017
</div>
Led small, high-impact teams to investigate and capture high-value (i.e. $10+ million) opportunities across North America. Collaborated with researchers around the world. Attained record-breaking product yield.

Initiated the use of deep learning and statistical modeling to evaluate and make predictions on large sets of noisy data. Proposed and implemented data-driven solutions that resolved long-standing issues in difficult environments.
<hr>
<h4 style="margin:0">
Machine Learning Engineer (Electrical Engineer)
</h4>
<div style="color:#a0aec0;margin-bottom:1.0em">
2014 &ndash; 2016
</div>
Launched the first deep learning and statistical machine learning equipment management programs. Developed software that cleaned and visualized data, trained machine learning models, and made accurate predictions on complex data. Supervised up to 14 technical personnel concurrently in safety-critical environments.
</div>
</div>
<hr>

<h2 style="margin-bottom:1.0em;font-weight:400;font-size:26px">Service</h2>

<div class="row">
<div class="col" markdown="1">
<span style="font-weight:500">Reviewer for NeurIPS 2020, ICML 2020</span>

<span style="font-weight:500">Designed a biogas plant to supply clean, renewable energy to villagers in rural India</span> &nbsp;
<span style="display:inline-block">
<a href="/assets/biogas.mp4" class="btn btn-primary btn-sm" style="margin-top:-1.0em">Video</a>
</span>
</div>
</div>
