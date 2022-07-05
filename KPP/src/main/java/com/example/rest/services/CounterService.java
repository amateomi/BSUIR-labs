package com.example.rest.services;

import com.example.rest.loggers.GlobalLogger;
import lombok.Getter;
import org.apache.logging.log4j.Level;
import org.springframework.stereotype.Service;

@Service
public class CounterService {

    @Getter
    private long count;

    public synchronized void increment() {
        ++count;
        GlobalLogger.Log(Level.INFO, "CounterService increment: count=" + count);
    }

}
