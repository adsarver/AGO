import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

public class tester {
    protected static final Logger log = LogManager.getLogger("[" + tester.class.getName() + "]");

    public static void main(String[] args) {
        System.out.println("Hello, World!");
        log.error("AHHH");
        log.info("thank goidness");
    }
}
