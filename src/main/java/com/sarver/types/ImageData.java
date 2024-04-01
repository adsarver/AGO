package com.sarver.types;

public class ImageData {
    private final String carMake;
    private final String carModel;
    private final String garageName;
    private final String personName;

    public ImageData(String carMake, String carModel, String garageName, String personName) {
        this.carMake = carMake;
        this.carModel = carModel;
        this.garageName = garageName;
        this.personName = personName;
    }

    public String getCarMake() {
        return carMake;
    }

    public String getCarModel() {
        return carModel;
    }

    public String getGarageName() {
        return garageName;
    }

    public String getPersonName() {
        return personName;
    }
}
