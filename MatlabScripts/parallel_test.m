function parallel_sparse_matrix_multiplication_test()
    sizesVector = [];
    time = [];
    procentMatrix = [];
    elementsR = [];

    % Matrix size
    size = 10000;

    fprintf('Running parallel performance test with matrix size %d...\n', size);

    parpool;  % Start parallel pool (if not already running)

    for n = 1:10
        sparsity = n / 100.0;
        fprintf('Sparsity test: %.2f%%\n', sparsity * 100);

        repeat = 1;
        averageTime = 0.0;
        nnzResult = 0;

        for k = 1:repeat
            % Generate random sparse matrices A and B
            fprintf('Generating sparse matrices with %.2f%% non-zero elements...\n', sparsity * 100);
            A = sprand(size, size, sparsity);
            B = sprand(size, size, sparsity);

            % Measure time for matrix multiplication in parallel
            tic;
            C = A * B;  % MATLAB will use optimized parallel operations
            timeTaken = toc;

            averageTime = averageTime + timeTaken;
            nnzResult = nnz(C);  % Count number of non-zero elements in result
        end

        averageTime = averageTime / repeat;

        sizesVector = [sizesVector; size];
        procentMatrix = [procentMatrix; n];
        elementsR = [elementsR; nnzResult];
        time = [time; averageTime];

        fprintf('Average time with procent nnz of %.2f%%: %.6f seconds\n', n, averageTime);
    end

    fprintf('-------------------------------------------------------------------------------------------\n');
    fprintf(' Size of A |    Time    | Procent of nnz in A matrix | Number of values in result matrix \n');
    fprintf('-------------------------------------------------------------------------------------------\n');

    for i = 1:length(sizesVector)
        fprintf('   %d   |  %.6f s  |             %d%%             |              %d\n', ...
                sizesVector(i), time(i), procentMatrix(i), elementsR(i));
    end

    fprintf('-------------------------------------------------------------------------------------------\n');
    delete(gcp);  % Shut down parallel pool
end
