import SwiftUI

struct AuthView: View {
    // Данные, которые пользователь вводит
    @State private var firstName: String = ""
    @State private var lastName: String = ""
    @State private var email: String = "" // Добавляем поле для почты
    @State private var password: String = ""
    
    // Для отображения статуса операции (например, успешного входа или ошибки)
    @State private var message: String = ""
    
    var body: some View {
        VStack(spacing: 20) {
            Text("зенов тащит")
                .font(.largeTitle)
                .padding(.bottom, 40)
            
            // Поля для ввода имени, фамилии, почты и пароля
            TextField("Имя", text: $firstName)
                .textFieldStyle(RoundedBorderTextFieldStyle())
                .autocapitalization(.none)
                .padding(.horizontal)
            
            TextField("Фамилия", text: $lastName)
                .textFieldStyle(RoundedBorderTextFieldStyle())
                .autocapitalization(.none)
                .padding(.horizontal)
            
            TextField("Email", text: $email)
                .keyboardType(.emailAddress)
                .textFieldStyle(RoundedBorderTextFieldStyle())
                .autocapitalization(.none)
                .padding(.horizontal)
            
            SecureField("Пароль", text: $password)
                .textFieldStyle(RoundedBorderTextFieldStyle())
                .padding(.horizontal)
            
            // Кнопка для входа
            Button("Войти") {
                login()
            }
            .padding()
            .background(Color.blue)
            .foregroundColor(.white)
            .cornerRadius(8)
            
            // Кнопка для регистрации
            Button("Зарегистрироваться") {
                register()
            }
            .padding()
            .background(Color.green)
            .foregroundColor(.white)
            .cornerRadius(8)
            
            // Ссылка "Забыли логин или пароль?"
            Button(action: {
                forgotPassword()
            }) {
                Text("Забыли логин или пароль?")
                    .foregroundColor(.blue)
                    .underline()
            }
            
            // Отображаем сообщение об успехе или ошибке
            Text(message)
                .foregroundColor(.red)
                .padding()
        }
        .padding()
    }
    
    // Функция для входа
    func login() {
        guard let url = URL(string: "http://127.0.0.1:8000/login") else {
            message = "Неверный URL"
            return
        }
        
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        
        // Устанавливаем данные в формате JSON
        let parameters: [String: Any] = [
            "firstName": firstName,
            "lastName": lastName,
            "email": email,
            "password": password
        ]
        request.httpBody = try? JSONSerialization.data(withJSONObject: parameters)
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        
        URLSession.shared.dataTask(with: request) { data, response, error in
            if let error = error {
                DispatchQueue.main.async {
                    message = "Ошибка: \(error.localizedDescription)"
                }
                return
            }
            
            if let data = data, let responseString = String(data: data, encoding: .utf8) {
                DispatchQueue.main.async {
                    message = responseString
                }
            }
        }.resume()
    }
    
    // Функция для регистрации
    func register() {
        guard let url = URL(string: "http://127.0.0.1:8000/register") else {
            message = "Неверный URL"
            return
        }
        
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        
        let parameters: [String: Any] = [
            "firstName": firstName,
            "lastName": lastName,
            "email": email, // Почта для регистрации
            "password": password
        ]
        request.httpBody = try? JSONSerialization.data(withJSONObject: parameters)
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        
        URLSession.shared.dataTask(with: request) { data, response, error in
            if let error = error {
                DispatchQueue.main.async {
                    message = "Ошибка: \(error.localizedDescription)"
                }
                return
            }
            
            if let data = data, let responseString = String(data: data, encoding: .utf8) {
                DispatchQueue.main.async {
                    message = responseString
                }
            }
        }.resume()
    }
    
    // Функция для восстановления логина или пароля
    func forgotPassword() {
        guard let url = URL(string: "http://127.0.0.1:8000/forgot-password") else {
            message = "Неверный URL"
            return
        }
        
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        
        // Отправляем почту для восстановления логина или пароля
        let parameters: [String: Any] = [
            "email": email // Почта для восстановления
        ]
        request.httpBody = try? JSONSerialization.data(withJSONObject: parameters)
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        
        URLSession.shared.dataTask(with: request) { data, response, error in
            if let error = error {
                DispatchQueue.main.async {
                    message = "Ошибка: \(error.localizedDescription)"
                }
                return
            }
            
            if let data = data, let responseString = String(data: data, encoding: .utf8) {
                DispatchQueue.main.async {
                    message = responseString
                }
            }
        }.resume()
    }
}

@main
struct firstApp: App {
    var body: some Scene {
        WindowGroup {
            AuthView()
        }
    }
}
