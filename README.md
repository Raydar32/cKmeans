# ImageReader
Lab assignment for Parallel Computing course (at) University of Florence.

### Features

- Parallel and sequential non-blocking image reader.
- Can open images (.jpg) from various sources.
- Can select the number of cores to use.
- Detailed report and presentation inside with informations and tutorial.

### Implementation details
Implemented in Java using the concurrency framework, the main paradigm is the "fork-join".
The main idea is to divide the work of importing images in tasks executed by threads.
An image is a Java object that loads a real .jpg image inside the program memory.
An accurate description of implementation is contained in report.pdf.

### Screenshot
![alt text](https://i.ibb.co/mF4fL6h/immagine.png)

### License
Licensed under the term of MIT License.
