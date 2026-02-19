const express = require('express');
const multer = require('multer');
const { PythonShell } = require('python-shell');
const path = require('path');
const fs = require('fs'); // Добавляем модуль fs

const app = express();

// Настраиваем хранилище для multer с сохранением оригинального имени
const storage = multer.diskStorage({
    destination: 'uploads/',
    filename: function (req, file, cb) {
        // Сохраняем оригинальное имя файла с расширением
        cb(null, file.originalname);
    }
});

const upload = multer({ 
    storage: storage, // Используем наше хранилище
    fileFilter: (req, file, cb) => {
        const ext = path.extname(file.originalname).toLowerCase();
        if (ext === '.csv' || ext === '.xlsx' || ext === '.xls') {
            cb(null, true);
        } else {
            cb(new Error('Only CSV and Excel files are allowed!'), false);
        }
    },
    limits: {
        fileSize: 10 * 1024 * 1024 // Ограничение 10MB
    }
});

// Настройки
app.set('views', path.join(__dirname, 'views'));
app.set('view engine', 'ejs');
app.use(express.static('public'));
app.use(express.json()); // Для парсинга JSON

// Создаем папки, если их нет
const uploadDir = path.join(__dirname, 'uploads');
const outputDir = path.join(__dirname, 'public', 'output');

[uploadDir, outputDir].forEach(dir => {
    if (!fs.existsSync(dir)) {
        fs.mkdirSync(dir, { recursive: true });
    }
});

// Маршруты
app.get('/', (req, res) => {
    res.render('index');
});

app.post('/analyze', upload.single('ecgFile'), async (req, res) => {
    try {
        if (!req.file) {
            return res.status(400).json({ 
                error: 'No file uploaded', 
                details: req.fileValidationError || 'Invalid file type' 
            });
        }

        const filePath = req.file.path;
        const outputPath = path.join(outputDir, `${req.file.filename}_result.json`);

        console.log("filePath: ", filePath)
        console.log("outputPath: ", outputPath)
        
        // Опции для Python скрипта
        const options = {
            mode: 'text',
            pythonOptions: ['-u'],
            scriptPath: path.join(__dirname, '..', 'prot2', 'ml'), // Путь к папке ml
            args: [filePath, outputPath]
        };
        console.log("options.scriptPath: ", options.scriptPath)

        console.log("Запуск Python скрипта");
        /*const results = await new Promise((resolve, reject) => {
            const pythonProcess = PythonShell.run('ecg_analysis.py', options, (err, results) => {
                if (err) {
                    console.error("Ошибка Python:", err);
                    reject(err);
                } else {
                    console.log("Python скрипт завершился, результаты:", results);
                    resolve(results);
                }
            });
        });*/

        try {
            const results = await new Promise((resolve, reject) => {
                // Таймаут для безопасности (например, 5 минут)
                const timeout = setTimeout(() => {
                    reject(new Error("Python скрипт превысил время выполнения"));
                }, 30000); // 5 минут в миллисекундах

                const pythonProcess = new PythonShell('ecg_analysis.py', options);
                let output = [];

                pythonProcess.on('message', (message) => {
                    console.log("Python stdout:", message);
                    output.push(message);
                });

                pythonProcess.on('stderr', (stderr) => {
                    console.error("Python stderr:", stderr);
                });

                pythonProcess.on('close', (code) => {
                    clearTimeout(timeout);
                    if (code !== 0) {
                        reject(new Error(`Python скрипт завершился с кодом ${code}`));
                    } else {
                        console.log("Python скрипт успешно завершился");
                        resolve(output);
                    }
                });

                pythonProcess.on('error', (err) => {
                    clearTimeout(timeout);
                    console.error("Ошибка Python:", err);
                    reject(err);
                });
            });

            console.log("Результаты Python скрипта:", results);
        } catch (error) {
            console.error("Ошибка при выполнении Python скрипта:", error);
            // Здесь можно добавить обработку ошибки
        } 
        console.log("Чтение результатов")
        const result = JSON.parse(fs.readFileSync(outputPath, 'utf8'));
        
        console.log("Добавляем URL для изображения")
        result.ecg_image_url = `/output/${path.basename(result.ecg_image)}`;
        
        console.log("Удаляем временный файл")
        fs.unlinkSync(filePath);
        
        //res.json(result);
        // Рендерим шаблон вместо отправки JSON
        res.render('results', { result });
    } catch (error) {
        console.error('Analysis error:', error);
        res.status(500).json({ 
            error: 'Analysis failed', 
            details: error.message 
        });
        
        // Удаляем временный файл в случае ошибки
        if (req.file && fs.existsSync(req.file.path)) {
            fs.unlinkSync(req.file.path);
        }
    }
});

// Обработка ошибок
app.use((err, req, res, next) => {
    console.error(err.stack);
    res.status(500).json({ error: 'Something went wrong!' });
});

// Запуск сервера
const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
    console.log(`Server running on port ${PORT}`);
});