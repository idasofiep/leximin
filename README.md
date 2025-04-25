# leximin
Master thesis project repository

Overview
========
This repository containts implementation of different leximin algorithms. The intention is to analyse and compare different leximin methods.

The main program and methods can be found in the leximin.py file.

There are two types of problems that can be solved: allocations and distributions. 


Algorithms
==========

### Ordered Outcomes Method
A general formula for finding leximin optimal solutions. 

Described here:
* Ogryczak and Sliwinśki [On Direct Methods for Lexicographic Min-Max Optimization](https://link.springer.com/chapter/10.1007/11751595_85)

The original multi-objective leximin problem is transformed to a lexicographic optimization problem, where N (number of agents) single objective programs are solved in turn and their result added to the optimization model. The method has been shown to be uneffective or non-usable for too large problems, but can be particularily useful for non-convex or continous problems that can not be solved by ordered values or saturation methods. This method is also possible to use in approximation leximin algorithms. 

lex min [t_1 + sum(d_1j), ... , t_m + sum(d_mj)]
s.t.     t_k + d_kj >= f_j(X)
         d_kj >= 0

After each iteration, set the solved objective as a new constraint and add its value as the upper or lower bound.

### Ordered Values Method
The ordered values method is, like the ordered outcomes, a redefinition of the original multi-objective problem to a lexicographic optimization problem. There new problem has V single-objective programs, where V is the number of potential values for all agents value functions. This means that there needs to be a finite set of possible values that is known or can be calculated to use this method.
 
 lex min [sum(h_2j), ... , sum(h_rj)]
 s.t.     h_kj >= f_j(X) - v_k
          h_kj >= 0

The running time for this method is far better than ordered outcomes, which can make it especially useful for larger problem sets. Experiments show that for large numbers (1000+) of possible values, the running time is still a lot faster than ordered outcomes. 

Another nice thing with this method is that you don'† necessarily need to know all possible values, as long as the set of values containt all possible values it does not matter if the set also contains values that are not part of the possible values set. So if you know that a problem instance have only integer function values between 0 and 200, but you don't know or find it to take too much resources to calculate the exact possible values, you can use all integers between 0 and 200 and the method will still give the correct result. 

### Saturation Method


For every free objective f_j :
Solve the following single-objective problem:

max f_j(x):
subject to x in X,
f_k(x) >= z_k for all saturated objectives k,
f_k(x) >= z_max for all free objectives k

If the optimal value equals z_max, then objective j becomes saturated from now on.
otherwise, the optimal value must be larger than z_max; objective j remains free for now. 



Running the program
===================
Some example data files are provided to run the different algorithms.

Testing the ordered outcomes method using the large allocation example:
```
$ python3 leximin.py large.csv oo

```

Testing the ordered values method with the large allocation example:
```
$ python3 leximin.py large.csv ov

```

Testing the saturation method using the large allocation example:
```
$ python3 leximin.py large.csv sa

```

Testing the saturation method using the large sortition example:
```
$ python3 leximin.py large.csv so

```


Citizens Assemblies
==========
Citizens Assemblies are interesting as there has been done a lot of research about it lately, and it's one of few real life examples I have found that use an implementation of leximin/leximax optimization. The general framework for generating citizen's assembly are described in the article "Fair Algorithms for selection Citizen's assemblies" (https://www.researchgate.net/publication/353697325_Fair_algorithms_for_selecting_citizens'_assemblies) 

The implementation of the framwork can be found here:
https://github.com/sortitionfoundation/stratification-app/tree/main

There is also a web resource for making citizen assembly lotteries, panelog.org, that is also made by the sortition foundation group.

I have made a simplified implementation of this method in the file citizensassemlies.py. The program should still produce equivalent results as the original program.

### The general method
The point with citizens assembliy panels is that they should be a mini-representation of the society as a whole. A citizens assembly panel can represent the general population in public questions and influence decition takers as a type of direct democracy. There is a lot of interesting political aspects regarding these types of direct democratic systems that are most likely not too relevant for this case. 

The process most commonly used to form panels for citizens assemblies is called sortition and is done in two steps. The first step is to make a large pool of possible participants. This can be done by sending invitation to a set of random citizens. After doing this first step, there should be a relaively large group of volunteers willing to participate. There is one important problem that makes the second step important; representativeness. The group of volunteers is most likely not a good representation of the population even if the invitations where sent to a random set of people; for instance will groups in the society that are more interested in politics be more likely to answer the invitation about participating in a political assembly. The population can be described using categories such as gender, age or political views. A population that has 49% indentifying as female, 49% as male and 2% as other gender should be represented by a minipublic with around the same ratio, even if the volunteers have a different gender ratio. 

The second step is supposed to reverse the biases made in the first step. This step is called stratification, and this is the part of the method that we will focus on. A number of possible panels are constructed that are considered "representative" based on some pre-defined criteria. Then, one final panel is choosen from all the possible panels by lottery. The point with using lottery to choose one of many acceptable panels, is to ensure fairness among participants, so that all volunteers have a change of being chosen.

### The Stratification Framework
The overall population are described using categories and specific quotas for each category. So for a panel of size 10, we can set the possible quota for female and male to between 4 and 6. (Side note: a possible improvement on the original framework is to use percentage ratio instead of number quotas so the category file input does not need to be edited for different panel sizes). The algorithm should produce a number of panels that respect all quotas and because of that are representative of the population. 

Then, we also need to ensure fairness among all agents, which means that all agents should have equal possibility of being chosen. The problem is that it is often not possible to both ensure representativeness and equal probability of being chosen. Underrepresented groups in the pool will have a greater change of being part of the final panels. The framework presented by Flanigal et al (place reference here) use a leximin strategy to ensure maximal fairness among agents. The way it works, is that all agents probabilities are leximin ordered so that the lottery produced gives all agents a leximax probability of being chosen. They argue that other methods of making stratification lotteries often exclude some agents from being chosen or give them equal to zero probability, and by applying leximin ordering of probabilities this is prevented. 

The framework use a saturation method for the leximin optimization, and column generation to find possible panels for the final lottery. For each iteration at least one agents probability will be saturated or a new panel is added to the solution.


