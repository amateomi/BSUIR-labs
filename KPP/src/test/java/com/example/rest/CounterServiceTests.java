package com.example.rest;

import com.example.rest.services.CounterService;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.stream.IntStream;

import static org.junit.jupiter.api.Assertions.assertEquals;

@SpringBootTest
public class CounterServiceTests {

    @Autowired
    CounterService counterService;

    @Test
    void testIncrement() {
        ExecutorService executorService = Executors.newFixedThreadPool(10);

        IntStream.range(0, 1000).forEach(count -> executorService.execute(counterService::increment));

        executorService.shutdown();
        try {
            if (!executorService.awaitTermination(800, TimeUnit.MILLISECONDS)) {
                executorService.shutdownNow();
            }
        } catch (InterruptedException e) {
            executorService.shutdownNow();
        }

        assertEquals(1000, counterService.getCount(), "Synchronization check");
    }

}
