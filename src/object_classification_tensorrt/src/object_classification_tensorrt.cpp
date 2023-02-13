#include <chrono>
#include <functional>
#include <memory>
#include <string>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"

using namespace std::chrono_literals;




class ClassificationPublisher : public rclcpp::Node{
    public:
        ClassificationPublisher() : Node("classification_publisher"), count_(0){
            publisher_ = this -> create_publisher<std_msgs::msg::String>("classification_topic", 10);
            timer_ = this->create_wall_timer(500ms, std::bind(&MinimalPublisher::timer_callback, this))
        }
    private:
    void timer_callback(){
        auto message = std_msgs::msg:String();
        message.data = "TeSt";
        RCLCPP_INFO(this->get_logger(), "Publishing: ");
        publisher_ -> publish(message);
    }
    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::Publisher<std_msgs::String>::SharedPtr publisher_;
    size_t count_;
}