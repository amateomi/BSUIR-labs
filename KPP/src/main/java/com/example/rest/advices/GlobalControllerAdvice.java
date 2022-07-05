package com.example.rest.advices;

import com.example.rest.loggers.GlobalLogger;
import org.apache.logging.log4j.Level;
import org.jetbrains.annotations.NotNull;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.http.converter.HttpMessageNotReadableException;
import org.springframework.web.bind.MissingServletRequestParameterException;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.RestControllerAdvice;
import org.springframework.web.method.annotation.MethodArgumentTypeMismatchException;

@RestControllerAdvice
public class GlobalControllerAdvice {

    @ExceptionHandler(MissingServletRequestParameterException.class)
    public ResponseEntity<Object> handleIllegalArgumentException(@NotNull MissingServletRequestParameterException e) {
        var message = "Status 400: missing param " + e.getParameterName();
        GlobalLogger.Log(Level.WARN, "GlobalControllerAdvice: " + message);
        return new ResponseEntity<>(message, HttpStatus.BAD_REQUEST);
    }

    @ExceptionHandler(MethodArgumentTypeMismatchException.class)
    public ResponseEntity<Object> handleTypeMismatchException(@NotNull MethodArgumentTypeMismatchException e) {
        var message = "Status 400: invalid param type, " + e.getName() + " must be " + e.getRequiredType();
        GlobalLogger.Log(Level.WARN, "GlobalControllerAdvice: " + message);
        return new ResponseEntity<>(message, HttpStatus.BAD_REQUEST);
    }

    @ExceptionHandler(HttpMessageNotReadableException.class)
    public ResponseEntity<Object> handleMessageNotReadableException(@NotNull HttpMessageNotReadableException e) {
        var message = "Status 400: invalid param type";
        GlobalLogger.Log(Level.WARN, "GlobalControllerAdvice: " + message + ": " + e.getMessage());
        return new ResponseEntity<>(message, HttpStatus.BAD_REQUEST);
    }

    @ExceptionHandler(Exception.class)
    public ResponseEntity<Object> handleServerException(@NotNull Exception e) {
        var message = "Status 500: internal server error";
        GlobalLogger.Log(Level.ERROR, "GlobalControllerAdvice: " + message + ": " + e.getMessage());
        return new ResponseEntity<>(message, HttpStatus.INTERNAL_SERVER_ERROR);
    }

}
