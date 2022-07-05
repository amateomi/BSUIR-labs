package com.example.rest.loggers;

import org.apache.logging.log4j.Level;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

public class GlobalLogger {
    private static final Logger logger = LogManager.getLogger(GlobalLogger.class);

    public static void Log(Level level, Object message) {
        logger.log(level, message);
    }

}
