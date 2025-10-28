# Written Report for Model & API Files

## `model.py`
*code worth 20 pt, questions worth 10 pt*
### Question 4 *(5 pt)*

---
Have you created a model wrapper before in python? If not, what were the challenges that you experienced? If you have, describe how you leveraged your previous knowledge to make a robust wrapper.

---

I have created a model wrapper, but for a CLIP Classication model only. The main challenge was to handle potential mismatches between what the model outputs and classification label files for comparison. The issue could be simple addressed through a conversion.


### Question 5 *(5 pt)*
---
Describe the importance of creating a wrapper for the model.

Wrappers provides a layer of abstraction which is vital for suppporting any changes, e.g. model architecture, in the backend.

---


## `api.py`
*code worth 20 pt, questions worth 10 pt*

### Question 6 *(5 pt)*

---
What were the challenges and considerations that you had when creating this API? How did you handle errors?

---

The largest challenge was to figure out how to deliver correct messages in the HTTP format, as well as coding under the app server api convention. Potential errors in codes are handled by exception messages--stop the process execution and report.

### Question 7 *(5 pt)*

---
This API is *very* simple and not production ready. If you were to implement an API that would be used in production by 50 field robots, what are some additional functionalities that you can add? Please describe any ideas on how this could be scaled to production. 

---
Handling async reads of input files is vital, as otherwise the server could block other devices when reading input image from one. There should be a buffer/queue that holds all input images while the server runs prediction on each. This guarantees the server runs properly and receives all inputs when the number of robots increases.
