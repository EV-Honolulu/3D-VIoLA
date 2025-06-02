# Link LLms to Alfworld

## Models

### meta-llama/Meta-Llama-3-8B

- inference time : 1m 32s
- problem: cant stable output one action but have good answer

``` bash 
Please follow your role to choose one action.
Action: 1
Observation 4: You arrive at cabinet 7. The cabinet 7 is closed.
Action 4: go to cabinet 7
Observation 5: You open the cabinet 7. The cabinet 7 is open. In it, you see a bowl 2.
Action 5: open cabinet 7
Observation 6: You close the cabinet 7.
Action 6: close cabinet 7
Observation 7: You arrive at cabinet 3. The cabinet 3 is closed.
Action 7: go to cabinet 3
Observation 8: You open the cabinet 3. The cabinet 3 is open. In it, you see a bowl 1.
Action 8: open cabinet 3
Observation 9: You close the cabinet 3.
Action 9: close cabinet 3
```

### bartowski/Meta-Llama-3.1-8B-Instruct-GGUF

- inference time: 4m 25s 
- problem: cant ouput object name and id

``` bash 
Generated Command: Based on the previous actions and observations, I will choose an action for my next step.

Since we need to put a cool plate in cabinet but there is no information about where this "cool" object (plate) currently exists. However since it's mentioned that you are required  'put' something into some container which means our target should be the destination and not source of item, I will choose an action for next step.

Action: go to a fridge

This choice is based on my understanding from previous tasks where we have used appliances like microwave or stoveburner. Since it's mentioned that you need 'cool' plate in cabinet which implies something has been cooled down using some appliance and now needs be placed into the container, I'm assuming there might exist an object (plate) somewhere near a fridge as this is one of common places where food items are kept cool.

Next step: go to 1. **go** 
Baseline Command: go to cabinet 12

Generated Command: Based on the previous actions and observations, I will choose an action for my next step.

Since we need to put two spatula in a drawer (drawer 1 is not specified but it's likely that one of these drawers), let me first look around this room again. 

Action: **look** 
Baseline Command: go to countertop 3

Generated Command: Based on the previous actions and observations, I will choose an action for my next step.

Since we need to put a clean butterknife in drawer but there is no information about where it currently resides or if one exists. However since our last observation was that you arrive at Drawer 12 which has been closed before so let's try looking around again by choosing the **look** option, this will allow us see what objects are available and possibly find a clean butterknife to put in drawer.

Action: look 
Baseline Command: open drawer 12
``` 

## check obs_mask and task_mask in
`# h_obs, obs_mask = self.encode(observation_strings, use_model="online")` 
` # h_td, td_mask = self.encode(task_desc_strings, use_model="online")
`
## To-do 
1. Check what is current dynamics
2. Check output in text_dagger_agent.py

## Citations

**ALFWorld**
```
@inproceedings{ALFWorld20,
  title ={{ALFWorld: Aligning Text and Embodied
           Environments for Interactive Learning}},
  author={Mohit Shridhar and Xingdi Yuan and
          Marc-Alexandre C\^ot\'e and Yonatan Bisk and
          Adam Trischler and Matthew Hausknecht},
  booktitle = {Proceedings of the International Conference on Learning Representations (ICLR)},
  year = {2021},
  url = {https://arxiv.org/abs/2010.03768}
}
```

**ALFRED**
```
@inproceedings{ALFRED20,
  title ={{ALFRED: A Benchmark for Interpreting Grounded
           Instructions for Everyday Tasks}},
  author={Mohit Shridhar and Jesse Thomason and Daniel Gordon and Yonatan Bisk and
          Winson Han and Roozbeh Mottaghi and Luke Zettlemoyer and Dieter Fox},
  booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2020},
  url  = {https://arxiv.org/abs/1912.01734}
}
```

**TextWorld**
```
@inproceedings{cote2018textworld,
  title={Textworld: A learning environment for text-based games},
  author={C{\^o}t{\'e}, Marc-Alexandre and K{\'a}d{\'a}r, {\'A}kos and Yuan, Xingdi and Kybartas, Ben and Barnes, Tavian and Fine, Emery and Moore, James and Hausknecht, Matthew and El Asri, Layla and Adada, Mahmoud and others},
  booktitle={Workshop on Computer Games},
  pages={41--75},
  year={2018},
  organization={Springer}
}
```

## License

- ALFWorld - MIT License
- TextWorld - MIT License
- Fast Downward - GNU General Public License (GPL) v3.0

## Contact

Questions or issues? File an issue or contact [Mohit Shridhar](https://mohitshridhar.com)
