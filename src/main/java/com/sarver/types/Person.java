package com.sarver.types;

import org.jetbrains.annotations.NotNull;

public class Person {
    private String firstName;
    private String lastName;
    private String name;

    public Person(String firstName, String lastName) {
        this.firstName = firstName;
        this.lastName = lastName;
        this.name = firstName + " " + lastName;
    }

    public String getName() {
        return name;
    }

    public String getFirstName() {
        return firstName;
    }

    public String getLastName() {
        return lastName;
    }

    public void setName(String name) {
        this.name = name;
    }

    public void setFirstName(String firstName) {
        this.firstName = firstName;
    }

    public void setLastName(String lastName) {
        this.lastName = lastName;
    }

    public boolean equals(@NotNull Person other) {
        return this.name.equals(other.getName());
    }

    public String toString() {
        return name;
    }
}
