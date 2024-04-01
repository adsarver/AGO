package com.sarver.controllers;

import java.sql.Connection;
import java.sql.DriverManager;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

public class DatabaseController {
    protected static final Logger log = LogManager.getLogger("[" + DatabaseController.class.getName() + "]");
    Connection con;

    public DatabaseController() {
        try {
            //Temporary fill ins
            Class.forName("oracle.jdbc.driver.OracleDriver");
            con = DriverManager.getConnection(
                    "jdbc:oracle:thin:@localhost:1521:orcl", "login1", "pwd1");
        } catch(Exception e) {
            log.error(e.getMessage());
        }
    }

    public void close() {
        try {
            con.close();
        } catch(Exception e) {
            log.error(e.getMessage());
        }
    }
}
