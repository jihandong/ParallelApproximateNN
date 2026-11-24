# ParallelApproximateNN
Final Project of Parallel Programming Algorithm, NYU Graduate Course

1. dataset preparation: unpack hdf5 and store train, test, answers into 3 data file(f32).
2. run single thread Faiss on local machine
3. impl checker
4. draft own algorithm & impl
    1. train
        1. Pick samples
        2. ⚡Pick initial centroid
        3. ⚡Lloydss
    2. ⚡add
    3. ⚡query
    3. integrate with main logic
5. test
    1. test single thread on local machine till pass
    2. test multi thread on remote machine
5. result visualization
6. (optional) optimization

## TODO
- Optimize the case of few threads
- Tradeoff:
    - Add id: 
    - Add vector: better for multi-query

