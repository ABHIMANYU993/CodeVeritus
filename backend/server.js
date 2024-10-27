const express = require('express');
const mongoose = require('mongoose');
const cors = require('cors');
const bcrypt = require('bcrypt');
const app = express();
const PORT = process.env.PORT || 5000;

// Middleware
app.use(cors());
app.use(express.json());

// MongoDB connection
mongoose.connect('mongodb://localhost:27017/ncode', { useNewUrlParser: true, useUnifiedTopology: true })
  .then(() => console.log('MongoDB connected'))
  .catch(err => console.error('MongoDB connection error:', err));

// User Schema
const userSchema = new mongoose.Schema({
  role: { type: String, required: true }, // 'user' or 'admin'
  username: { type: String, required: true, unique: true },
  email: { type: String, required: true, unique: true },
  password: { type: String, required: true },
});

const User = mongoose.model('User', userSchema);

// User Code Schema (with username included)
const UserCodeSchema = new mongoose.Schema({
  username: { type: String, required: true },  // Username of the user
  userId: {
    type: String,
    required: true, // Make it required if you always expect it
    unique: true, // Ensure userId is unique if necessary
      },    
  codes: {
    type: [String],  // Array of strings to store code snippets
    default: [],
  },
  submittedAt: { type: String, default: Date().toLocaleString("en-US", { timeZone: "Asia/Kolkata" }) },
  processed: { type: Boolean, default: false },
  processedAt: { type: Date, default: null },
});

const UserCode = mongoose.model('UserCode', UserCodeSchema);

// API: Submit or update user codes
app.post('/api/backend', async (req, res) => {
  try {
    const { userId, codes } = req.body;
    // Ensure userId and codes are provided
    if (!userId || !codes || !codes.length) {
      return res.status(400).json({ message: 'userId and codes are required.' });
    }

    // // Update or insert user codes by username
    // const updatedUserCode = await UserCode.updateOne(
    //   { userId },
    //   {
    //     $addToSet: { codes: { $each: codes } }, // Add codes to the array, avoiding duplicates
    //     submittedAt: new Date(),
    //     processed: false,
    //     processedAt: null,
    //   },
    //   { upsert: true }
    // );
    // Find the user code by userId and update it with the new code submission
    const updatedCode = await UserCode.findOneAndUpdate(
      { userId: userId }, // Find document by userId
      { codes: codes, submittedAt: new Date().toLocaleString("en-IN", { timeZone: "Asia/Kolkata" }),processed: false, }, // Update with new code array and timestamp
      { upsert: true, new: true, strict: true } // Create if not exists, return updated document
    );

    res.status(201).json({ message: 'User codes saved successfully!' });
  } catch (error) {
    console.error('Error saving user codes:', error);
    res.status(500).json({ message: 'Error saving user codes' });
  }
});

// API: Get all submitted user codes (including usernames)
app.get('/api/backend', async (req, res) => {
  try {
    const userCodes = await UserCode.find().sort({ submittedAt: -1 }).populate('username'); // Fetch all user codes
    res.json(userCodes);
  } catch (error) {
    console.error('Error fetching user codes:', error);
    res.status(500).json({ message: 'Error fetching user codes' });
  }
});

// API: User Signup
app.post('/api/signup', async (req, res) => {
  const { role, username, email, password } = req.body;

  try {
    // Check if username or email exists
    const existingUser = await User.findOne({ $or: [{ username }, { email }] });
    if (existingUser) return res.status(400).json({ message: 'User already exists' });

    // Hash password and save new user
    const hashedPassword = await bcrypt.hash(password, 10);
    const newUser = new User({ role, username, email, password: hashedPassword });

    await newUser.save();
    res.status(201).json({ message: `${role} registered successfully` });
  } catch (error) {
    console.error('Signup error:', error);
    res.status(500).json({ message: 'Server error' });
  }
});

// API: User Login
app.post('/api/login', async (req, res) => {
  const { role, username, password } = req.body;

  try {
    // Find user by username and role
    const user = await User.findOne({ username, role });
    if (!user) return res.status(400).json({ message: 'Invalid credentials' });

    // Compare password
    const isMatch = await bcrypt.compare(password, user.password);
    if (!isMatch) return res.status(400).json({ message: 'Invalid credentials' });

    res.json({ message: `${role} login successful`, username: user.username });
  } catch (error) {
    console.error('Login error:', error);
    res.status(500).json({ message: 'Server error' });
  }
});

// Start server (only one call to app.listen)
app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});
